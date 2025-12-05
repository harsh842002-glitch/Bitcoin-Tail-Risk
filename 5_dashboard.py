import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto Tail-Risk Engine", layout="wide")

# --- LOAD MODEL ---
try:
    with open('risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Error: 'risk_model.pkl' not found.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Scenario Settings")

portfolio_size = st.sidebar.number_input("Portfolio Value ($)", value=10000, step=1000)

# 1. Sentiment
fear_input = st.sidebar.slider("Fear & Greed Index", 0, 100, 50)

# 2. Leverage (Funding)
fund_bps = st.sidebar.slider("Funding Rate (Basis Points)", 0, 100, 10)
fund_input = fund_bps / 10000.0 

# 3. Liquidity (The New Variable)
# We use Standard Deviations (Sigma) for the slider to make it understandable
st.sidebar.subheader("💧 Liquidity Conditions")
illiq_sigma = st.sidebar.select_slider(
    "Market Condition",
    options=[-1, 0, 1, 2, 3, 4, 5],
    value=0
)
st.sidebar.caption("0 = Normal. 5 = Dried Up (Fragile).")

# --- PREDICTION ---
# We must match the EXACT column names from the model training
input_data = pd.DataFrame({
    'Intercept': [1.0], 
    'Fear_Lag1': [fear_input],
    'Fund_Lag1': [fund_input],
    'Illiq_Lag1': [illiq_sigma]  # <--- THIS LINE WAS LIKELY MISSING
})

predicted_return = model.predict(input_data)[0]
predicted_loss = portfolio_size * predicted_return


predicted_return = model.predict(input_data)[0]
predicted_loss = portfolio_size * predicted_return

# --- DISPLAY ---
st.title("📉 Bitcoin Tail-Risk Engine (Enhanced Microstructure)")
st.markdown("Quantifying the **5% Worst-Case Scenario** using Sentiment, Leverage, and Liquidity.")

col1, col2, col3 = st.columns(3)

# METRIC 1: VaR
with col1:
    st.subheader("⚠️ 1-Day VaR (95%)")
    color = "red" if predicted_return < -0.07 else "green"
    st.markdown(f"<h1 style='color:{color}'>{predicted_return:.2%}</h1>", unsafe_allow_html=True)

# METRIC 2: Loss
with col2:
    st.subheader("💰 Projected Loss")
    st.metric("Worst Case", f"${predicted_loss:,.2f}")

# METRIC 3: Risk Drivers
with col3:
    st.subheader("📊 Risk Factors")
    if fund_bps > 30:
        st.warning("High Leverage Risk detected!")
    elif illiq_sigma > 2:
        st.warning("Liquidity Crunch detected!")
    else:
        st.success("Market Microstructure Stable")

# --- CHART: Interactive Stress Test ---
st.divider()
st.subheader("🔥 Stress Test: The 'Death Spiral'")
st.markdown("What happens if **Leverage Explodes** while **Liquidity Dries Up**?")

# Generate 3D Surface Data for Plotly
x_fund = np.linspace(0, 0.01, 20)  # Funding Rates
y_illiq = np.linspace(0, 5, 20)    # Liquidity
z_risk = []

for f in x_fund:
    row = []
    for i in y_illiq:
        # Calculate Risk for every combination
        val = model.params['Intercept'] + \
              (model.params['Fear_Lag1'] * fear_input) + \
              (model.params['Fund_Lag1'] * f) + \
              (model.params['Illiq_Lag1'] * i)
        row.append(val)
    z_risk.append(row)

# 3D Plot
fig = go.Figure(data=[go.Surface(z=z_risk, x=x_fund*10000, y=y_illiq)])
# ... (Your existing loops that calculate z_risk are above here) ...

# 1. Create the Surface Map
fig = go.Figure(data=[go.Surface(z=z_risk, x=x_fund*10000, y=y_illiq)])

# --- PASTE THIS NEW BLOCK HERE ---
# Add a "You Are Here" Red Dot
fig.add_trace(go.Scatter3d(
    x=[fund_bps],    # User's Funding Input (X-axis)
    y=[illiq_sigma], # User's Liquidity Input (Y-axis)
    z=[predicted_return], # The Calculated Risk (Z-axis)
    mode='markers',
    marker=dict(size=12, color='red', symbol='diamond', line=dict(color='white', width=2)),
    name='Your Portfolio'
))
# ---------------------------------

# 2. Update Layout (Your existing code)
fig.update_layout(
    title="3D Risk Topology (Red Diamond = You)",
    scene=dict(
        xaxis_title='Funding (bps)',
        yaxis_title='Illiquidity (Sigma)',
        zaxis_title='Predicted VaR'
    ),
    template="plotly_dark",
    height=500
)

# 3. Show Chart (Your existing code)
st.plotly_chart(fig, use_container_width=True)
fig.update_layout(
    title="3D Risk Topology",
    scene=dict(
        xaxis_title='Funding (bps)',
        yaxis_title='Illiquidity (Sigma)',
        zaxis_title='Predicted VaR'
    ),
    template="plotly_dark",
    height=500
)
st.plotly_chart(fig, use_container_width=True)