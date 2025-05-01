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
    returns_file = "outputs/output_yearlywfilepath.csv"
    if os.path.exists(returns_file):
        df = pd.read_csv(returns_file)
        df['Date'] = pd.to_datetime(df['Year'])
        df.set_index('Date', inplace=True)
        return df
    else:
        st.error("output_yearly.csv not found.")
        st.stop()

try:
    returns_df = load_returns_data()
except Exception as e:
    st.error(f"Failed to load firm return data: {e}")
    st.stop()

# --- Load CEO Data ---
try:
    ceo_file = "outputs/output_yearly_with_image_paths.csv"
    ceo_df = pd.read_csv(ceo_file)
    ceo_df.columns = ceo_df.columns.str.strip()
    ceo_df['Ticker'] = ceo_df['Ticker'].str.strip().str.upper()
    ceo_df['Year'] = pd.to_numeric(ceo_df['Year'], errors='coerce').astype('Int64')
    ceo_df = ceo_df[(ceo_df['Year'] >= 2010) & (ceo_df['Year'] <= 2019)]
except Exception as e:
    st.error(f"Failed to load CEO data: {e}")
    st.stop()

# --- Full-width CEO and Company Info ---
st.subheader("View attributes by Company")

# Dropdowns
all_tickers = sorted(ceo_df['Ticker'].unique())
selected_company = st.selectbox("Please select company", all_tickers)

allowed_years = list(range(2010, 2020))
selected_year = st.selectbox("Please select year", allowed_years)

# Filter data based on selection
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

        firm_return = selected_data.get('Tenure_Cum_Ret_Overall')
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
def plot_cumulative_returns_by_ceo(ticker, ceo_name, ceo_df, returns_df):
    # Step 1: Find all years this CEO served at the selected company
    tenure_years = ceo_df[
        (ceo_df['Ticker'] == ticker) & 
        (ceo_df['CEO'] == ceo_name)
    ]['Date'].dropna().astype(int).tolist()

    if not tenure_years:
        st.warning("No years found for this CEO and company.")
        return None

    # Step 2: Create date ranges for each year
    all_dates = []
    for year in tenure_years:
        year_dates = returns_df.index[
            (returns_df.index.year == year)
        ]
        all_dates.extend(year_dates)

    if not all_dates:
        st.warning("No return data for selected CEO's tenure.")
        return None

    # Step 3: Filter return data
    plot_df = returns_df.loc[sorted(set(all_dates))].copy()

    if ticker not in plot_df.columns or 'SPY' not in plot_df.columns:
        st.error(f"Return data not available for {ticker} or SPY.")
        return None

    # Step 4: Calculate cumulative returns
    plot_df[f'Cum_Return_{ticker}'] = (1 + plot_df[ticker]).cumprod() - 1
    plot_df['Cum_Return_SPY'] = (1 + plot_df['SPY']).cumprod() - 1

    # Step 5: Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x=plot_df.index, y=f'Cum_Return_{ticker}', ax=ax, label=ticker)
    sns.lineplot(data=plot_df, x=plot_df.index, y='Cum_Return_SPY', ax=ax, label='SPY')

    plt.title(f"Cumulative Returns for {ceo_name} at {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

if selected_data is not None:
    ceo_name = selected_data.get('CEO')
    if ceo_name and selected_company:
        fig = plot_cumulative_returns_by_ceo(selected_company, ceo_name, ceo_df, returns_df)
        if fig:
            st.pyplot(fig)
    else:
        st.info("CEO or company info missing.")
