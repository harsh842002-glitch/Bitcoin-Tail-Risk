# Bitcoin-Tail-Risk
Overview
This repository contains a quantitative risk management framework that investigates the tail risk dynamics of Bitcoin. Traditional Value at Risk (VaR) frameworks frequently fail in cryptocurrency markets because they rely on Gaussian normality assumptions, leaving portfolios exposed to the "leptokurtic" (fat-tailed) reality of digital assets.  

To solve this, this project moves beyond standard Ordinary Least Squares (OLS) and exogenous macro factors. It utilizes Microstructure-Augmented Quantile Regression to directly model the conditional 5th percentile (τ=0.05) of returns, quantifying true "crash risk". The model is deployed as a live, Python-based Tail-Risk Engine via Streamlit, allowing risk managers to visualize the non-linear "Death Spiral" interactions between sentiment, leverage, and market fragility.  

The Core Hypothesis: Endogenous Liquidity Cascades
Cryptocurrency crashes are rarely unpredictable "Black Swan" events; they are often mechanical failures driven by internal market plumbing. This model ingests three key microstructure variables:  

Amihud Illiquidity Proxy: Measures market fragility and the cost of executing a trade (price impact per unit of volume).  

Perpetual Funding Rates: Acts as a proxy for the cost of leverage and speculative positioning in offshore derivatives markets.  

Fear & Greed Index: Quantifies behavioral sentiment and "noise trader risk".  

Key Empirical Finding: The "Funding Rate Flip"

The model empirically validates the "Long Squeeze" hypothesis. Under standard OLS models, high funding rates show a positive correlation with returns (+12.5), acting as a signal for bull market momentum. However, when targeting the 5th percentile and controlling for evaporating liquidity, the Quantile Regression coefficient flips to a statistically significant negative value (-6.35). This mathematically proves that leverage becomes toxic specifically when market depth degrades.  

Mathematical Formulation
To accurately capture the 95% Value at Risk, the model employs a specialized optimization function that fundamentally shifts how prediction errors are penalized. The algorithm minimizes a weighted sum of absolute deviations, applying a 95% penalty to under-predictions.  



By heavily penalizing the failure to predict a crash, the regression line is forced to anchor itself to the lower boundary (the "floor") of the data cloud, capturing structural breaks characteristic of liquidation cascades.  


System Architecture & Data Pipeline
The analytical architecture is built upon a high-fidelity dataset (2020–2025) and follows a standard Machine Learning pipeline.  


ETL Layer: Python scripts (utilizing pandas) clean and merge 24/7 crypto market data with traditional 5-day macro indices.  


Price & Volume: Ingested via the yfinance API.  

Derivatives Data: Aggregated perpetual futures funding rates from offshore venues (Binance, Bybit) via Coinglass.  


Sentiment: Ingested programmatically via the Alternative.me API.  


Model Layer: Serialized .pkl files encapsulate pre-trained Quantile Regression weights for instant inference.  

Presentation Layer: An interactive Streamlit frontend for dynamic scenario testing.  

Features
Real-Time Stress Testing: Manipulate parameters (VIX, Funding Rates, Liquidity) to simulate hypothetical "What-If" market conditions and instantly recalculate the predicted VaR.  

3D Risk Topology: A dynamic surface plot mapping the interaction between Leverage and Liquidity against Tail Risk, acting as an early warning system for the "Death Spiral" zone.  
