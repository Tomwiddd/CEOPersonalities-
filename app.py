import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("Study of CEO Headshot Attributes and Firm Returns")

# --- Load Returns Data ---
@st.cache_data
def load_returns_data():
    returns_file = "inputs/firm_returns.csv"
    if os.path.exists(returns_file):
        df = pd.read_csv(returns_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    else:
        st.error("firm_returns.csv not found.")
        st.stop()

try:
    returns_df = load_returns_data()
except Exception as e:
    st.error(f"Failed to load firm return data: {e}")
    st.stop()

# --- Load CEO Data ---
try:
    ceo_file = "ceo_face_analysis.csv"
    ceo_df = pd.read_csv(ceo_file)
    ceo_df.columns = ceo_df.columns.str.strip()
    ceo_df['Ticker'] = ceo_df['Ticker'].str.strip().str.upper()
    ceo_df['Year'] = pd.to_numeric(ceo_df['Year'], errors='coerce').astype('Int64')
    ceo_df = ceo_df[(ceo_df['Year'] >= 2010) & (ceo_df['Year'] <= 2019)]
except Exception as e:
    st.error(f"Failed to load CEO data: {e}")
    st.stop()

# --- Correlation Data ---
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
        return 'color: red'
    elif val > 0.02:
        return 'color: lightcoral'
    elif val < -0.05:
        return 'color: blue'
    elif val < -0.01:
        return 'color: lightblue'
    else:
        return 'color: black'

# --- Layout Start ---
col1, col2 = st.columns([1, 2])

# --- Column 1: Correlation Table ---
with col1:
    st.subheader("Correlation between image attributes and firm return")
    styled_corr = correlation_df.style.applymap(color_correlation).format("{:.3f}")
    st.dataframe(styled_corr)

# --- Column 2: Company & Year Select + CEO Info ---
with col2:
    st.subheader("View attributes by Company")

    all_tickers = sorted(ceo_df['Ticker'].dropna().str.strip().str.upper().unique())
    selected_company = st.selectbox("Please select company", all_tickers)

    allowed_years = list(range(2010, 2020))
    selected_year = st.selectbox("Please select year", allowed_years)

    filtered_data = ceo_df[
        (ceo_df['Ticker'] == selected_company) &
        (ceo_df['Year'] == selected_year)
    ]

    if filtered_data.empty:
        st.warning("No data available for this company and year.")
        selected_data = None
    else:
        selected_data = filtered_data.iloc[0]

        info_col, img_col = st.columns([1, 1])

        with img_col:
            image_path = selected_data.get('Image Path')
            if pd.notna(image_path) and os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.info("Image not available for this CEO/year.")
            firm_return = selected_data.get('Firm Return')
            if pd.notna(firm_return):
                st.metric(label="Firm Return", value=f"{firm_return:.1f}")
            else:
                st.metric(label="Firm Return", value="N/A")

        with info_col:
            st.text(f"Company: {selected_data.get('Ticker', 'N/A')}")
            st.text(f"Year: {selected_data.get('Year', 'N/A')}")
            st.text(f"CEO: {selected_data.get('CEO', 'N/A')}")
            st.text(f"Sex: {selected_data.get('dominant_gender', 'N/A')}")
            st.text(f"Race (inferred): {selected_data.get('dominant_race', 'N/A')}")
            st.text(f"Age: {selected_data.get('age', 'N/A')}")
            st.text(f"Dominant Emotion: {selected_data.get('dominant_emotion', 'N/A')}")
            st.text(f"Angry: {selected_data.get('angry', 0):.2f}")
            st.text(f"Disgust: {selected_data.get('disgust', 0):.2f}")
            st.text(f"Fear: {selected_data.get('fear', 0):.2f}")
            st.text(f"Happy: {selected_data.get('happy', 0):.2f}")
            st.text(f"Sad: {selected_data.get('sad', 0):.2f}")
            st.text(f"Surprise: {selected_data.get('surprise', 0):.2f}")
            st.text(f"Neutral: {selected_data.get('neutral', 0):.2f}")

# --- Cumulative Returns Plot ---
if selected_data is not None:
    st.subheader(f"Cumulative Returns During {selected_data.get('CEO', 'CEO')}'s Tenure")

    def plot_cumulative_returns(ticker, start_date, returns_df):
        plot_df = returns_df.copy()

        try:
            if start_date in plot_df.index:
                plot_df = plot_df[plot_df.index >= start_date]
            else:
                plot_df = plot_df[plot_df.index >= plot_df.index[plot_df.index > start_date][0]]
        except:
            st.warning("Start date not found in index.")
            return None

        if ticker not in plot_df.columns or 'SPY' not in plot_df.columns:
            st.error(f"Return data not available for {ticker} or SPY.")
            return None

        plot_df[f'Cum_Return_{ticker}'] = (1 + plot_df[ticker]).cumprod() - 1
        plot_df['Cum_Return_SPY'] = (1 + plot_df['SPY']).cumprod() - 1

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=plot_df, x=plot_df.index, y=f'Cum_Return_{ticker}', ax=ax, label=ticker)
        sns.lineplot(data=plot_df, x=plot_df.index, y='Cum_Return_SPY', ax=ax, label='SPY')

        plt.title(f"Cumulative Returns: {ticker} vs SPY")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    tenure_start = selected_data.get('Tenure Start')
    if pd.notna(tenure_start):
        if selected_company in returns_df.columns and 'SPY' in returns_df.columns:
            fig = plot_cumulative_returns(selected_company, tenure_start, returns_df)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("Ticker or SPY not in return data.")
    else:
        st.info("No start date available for this CEO.")
