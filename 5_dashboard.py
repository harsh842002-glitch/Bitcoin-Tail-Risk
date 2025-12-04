import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Crypto Tail-Risk Engine", layout="wide")

# --- LOAD THE TRAINED AI MODEL ---
try:
    with open('risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Error: 'risk_model.pkl' not found. Run '4_model.py' first!")
    st.stop()

# --- TITLE & SIDEBAR ---
st.title("📉 Bitcoin Tail-Risk Engine (Quantile AI)")
st.markdown("""
This tool uses **Quantile Regression (95% VaR)** to predict the "Worst Case Scenario" for Bitcoin.
It analyzes **Market Sentiment** and **Derivatives Leverage**.
""")

st.sidebar.header("🎛️ Stress Test Scenarios")

# USER INPUTS
# 1. Portfolio Size
portfolio_size = st.sidebar.number_input("Portfolio Value (₹)", value=100000, step=10000)

# 2. Fear & Greed Slider (0=Extreme Fear, 100=Extreme Greed)
# Default is 50 (Neutral)
fear_input = st.sidebar.slider("Crypto Fear & Greed Index", min_value=0, max_value=100, value=50)

# 3. Funding Rate Slider (in basis points)
# We let user pick "0.01%" but we convert it to real number (0.0001) for the math
fund_bps = st.sidebar.slider("Funding Rate (Basis Points)", min_value=0, max_value=100, value=10)
fund_input = fund_bps / 10000.0 # Convert 10 bps -> 0.0010

# --- THE AI PREDICTION ---
# We must feed the data into the model EXACTLY like we did in Day 4
# The column names must match: ['Fear_Lag1', 'Fund_Lag1', 'Intercept']
input_data = pd.DataFrame({
    'Intercept': [1.0], 
    'Fear_Lag1': [fear_input],
    'Fund_Lag1': [fund_input]
})

# Get the predicted return (e.g., -0.05)
predicted_return = model.predict(input_data)[0]
predicted_loss = portfolio_size * predicted_return

# --- DISPLAY METRICS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚠️ 1-Day VaR (95%)")
    # Color logic: If loss is worse than -7%, show RED
    if predicted_return < -0.07:
        st.markdown(f"<h1 style='color:red'>{predicted_return:.2%}</h1>", unsafe_allow_html=True)
        st.write("CRITICAL CRASH RISK")
    else:
        st.markdown(f"<h1 style='color:green'>{predicted_return:.2%}</h1>", unsafe_allow_html=True)
        st.write("Risk Within Normal Limits")

with col2:
    st.subheader("💰 Potential Loss")
    st.metric(label="Worst Case Tomorrow", value=f"₹ {predicted_loss:,.2f}")

with col3:
    st.subheader("📊 Market Regime")
    if fear_input < 20:
        st.info("Regime: 🐻 EXTREME FEAR")
    elif fear_input > 75:
        st.info("Regime: 🐂 EXTREME GREED")
    else:
        st.info("Regime: ⚖️ NEUTRAL")

# --- VISUALIZATION: The "What-If" Analysis ---
st.divider()
st.subheader("📈 Scenario Analysis: How Sentiment Impacts Risk")

# We generate a fake chart showing "If Fear drops to 0, what happens to Risk?"
x_axis = np.arange(0, 100, 5) # 0, 5, 10 ... 100
y_axis = []

# Calculate predicted risk for every level of fear
for f in x_axis:
    # Use the model's coefficients manually to plot the line
    # VaR = Intercept + (Coef_Fear * Fear) + (Coef_Fund * Current_Fund_Input)
    val = model.params['Intercept'] + (model.params['Fear_Lag1'] * f) + (model.params['Fund_Lag1'] * fund_input)
    y_axis.append(val)

# Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines+markers', name='Predicted VaR'))
fig.update_layout(
    title="Sensitivity Analysis: Risk vs. Sentiment",
    xaxis_title="Fear & Greed Index (Higher = Greedier)",
    yaxis_title="Predicted Worst-Case Return (VaR)",
    template="plotly_dark"
)
