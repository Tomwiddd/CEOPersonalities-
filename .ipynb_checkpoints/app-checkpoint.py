import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

page = st.sidebar.selectbox("Choose a page", ["Project Overview", "CEO Attributes", "Analysis"])

if page == "Project Overview":
    st.title("📊 Project Overview")
    st.write("Welcome to the CEO Analysis Dashboard.")

elif page == "CEO Attributes":
    st.title("🧠 CEO Attributes")
    st.write("This section explores CEO traits, clusters, etc.")

elif page == "Analysis":
    st.title("📈 Analysis")
    st.write("Visualize returns and evaluate CEO performance.")


# --- Page Config ---
st.set_page_config(layout="wide")
st.title("Study of CEO Headshot Attributes and Firm Returns")

# Custom CSS for white background
st.markdown("""
    <style>
        .stApp {
            background-color: black;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


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
# --- Check and Load Daily Returns CSV ---
try:
    returns_file = "outputs/output_daily.csv"
    r_df = pd.read_csv(returns_file)
    r_df.columns = r_df.columns.str.strip()
    r_df['Ticker'] = r_df['Ticker'].str.strip().str.upper()
    r_df['Year'] = pd.to_numeric(r_df['Year'], errors='coerce').astype('Int64')
    r_df['Date'] = pd.to_datetime(r_df['Date'], errors='coerce')
except Exception as e:
    st.error(f"Failed to load daily returns data: {e}")
    st.stop()

# --- Ensure required inputs are defined ---
if 'selected_data' in locals() and selected_data is not None:
    ceo_name = selected_data.get('CEO')
else:
    st.error("selected_data is not defined or is None.")
    st.stop()

if 'selected_company' not in locals():
    st.error("selected_company is not defined.")
    st.stop()

# --- Function: Plot cumulative returns during CEO's tenure ---
def plot_cumulative_returns_by_ceo(ticker, ceo_name, r_df):
    # Ensure 'Date' column is datetime
    r_df['Date'] = pd.to_datetime(r_df['Date'], errors='coerce')

    # Filter rows for the selected CEO and ticker
    ceo_data = r_df[(r_df['Ticker'] == ticker) & (r_df['CEO'] == ceo_name)].copy()

    if ceo_data.empty:
        st.warning("No data available for the selected CEO and company.")
        return None

    # Filter for CEO's tenure dates (2010-01-01 to 2019-12-31)
    mask_dates = (r_df['Date'] >= '2010-01-01') & (r_df['Date'] <= '2019-12-31')
    r_df = r_df[mask_dates]

    # Set date as index for time series operations
    r_df = r_df.set_index('Date')

    # Get firm data during this period
    firm_df = r_df[(r_df['Ticker'] == ticker) & (r_df['CEO'] == ceo_name)].copy()

    # Get SPY data for same date range
    spy_df = r_df[(r_df['Ticker'] == 'SPY') & (r_df.index.isin(firm_df.index))].copy()

    if firm_df.empty:
        st.warning("Company return data not available for CEO's tenure.")
        return None

    # Sort by date
    firm_df = firm_df.sort_index()
    firm_df['Return'] = firm_df['Return'].astype(float)
    cum_firm = (1 + firm_df['Return']).cumprod() - 1

    plot_spy = not spy_df.empty
    if plot_spy:
        spy_df = spy_df.sort_index()
        spy_df['Return'] = spy_df['Return'].astype(float)
        cum_spy = (1 + spy_df['Return']).cumprod() - 1

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=cum_firm.index, y=cum_firm, ax=ax, label=ticker)

    if plot_spy:
        sns.lineplot(x=cum_spy.index, y=cum_spy, ax=ax, label='SPY')

    plt.title(f"Cumulative Returns for {ceo_name} at {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig



# --- Safely run the plot ---
fig = plot_cumulative_returns_by_ceo(selected_company, ceo_name, r_df)
if fig:
    st.pyplot(fig)
