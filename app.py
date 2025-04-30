import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading ---

# Load monthly returns data
@st.cache_data
def load_returns_data():
    returns_file = "inputs/firm_return.csv" 
    if os.path.exists(returns_file):
        df = pd.read_csv(returns_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    else:
        st.warning(f"File not found: {returns_file}. Generating dummy data.")
        date_range = pd.date_range(start='2010-01-01', end='2025-12-31', freq='Y')
        tickers = ['AAPL', 'META', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
        np.random.seed(42)
        market_returns = np.random.normal(0.008, 0.04, len(date_range))
        all_returns = {}
        for ticker in tickers:
            if ticker == 'SPY':
                all_returns[ticker] = market_returns
            else:
                beta = np.random.uniform(0.8, 1.5)
                stock_returns = beta * market_returns + np.random.normal(0.002, 0.08, len(date_range))
                all_returns[ticker] = stock_returns
        df = pd.DataFrame(all_returns, index=date_range)
        return df

# Load CEO data
ceo_file = "ceo_face_analysis.csv"
if os.path.exists(ceo_file):
    ceo_df = pd.read_csv(ceo_file)
    ceo_df['Year'] = pd.to_datetime(ceo_df['Year'])
else:
    st.error(f"CEO data file not found: {ceo_file}")
    st.stop()

# --- Pseudo Data Creation ---

correlation_data = {
    'Attribute': [
        'LM Positive', 'LM Negative', 'ML Positive', 'ML Negative',
        'Risk Positive', 'Risk Negative', 'High Level Positive',
        'High Level Negative', 'Finance Positive', 'Finance Negative'
    ],
    'Correlation (ret_0)': [
        -0.089, -0.014, 0.024, 0.038, -0.013,
        -0.010, -0.093, -0.020, 0.035, 0.070
    ]
}
correlation_df = pd.DataFrame(correlation_data).set_index('Attribute')

def color_correlation(val):
    if val > 0.05:
        color = 'red'
    elif val > 0.02:
        color = 'lightcoral'
    elif val < -0.05:
        color = 'blue'
    elif val < -0.01:
        color = 'lightblue'
    else:
        color = 'black'
    return f'color: {color}'

st.title("Study of CEO Headshot Attributes and Firm Returns")  # Make sure this runs



# --- How to Run ---
# 1. Save the code above as a Python file (e.g., `ceo_app.py`).
# 2. Make sure you have streamlit, pandas, matplotlib and seaborn installed:
#    pip install streamlit pandas matplotlib seaborn
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the app using:
#    streamlit run ceo_app.py
