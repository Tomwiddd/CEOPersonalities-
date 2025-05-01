import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page selector
page = st.sidebar.selectbox("Choose a page", ["Project Overview", "CEO Attributes", "Analysis"])

if page == "Project Overview":
    st.title("📊 CEO Attributes and Firm Returns: Project Overview")
    st.write("This project explores how a CEO's headshot may correlate to a firm's returns. We looked at publically traded technology firms listed on the S&P 500 and their CEO's from 2010 - 2019. We analyzed pictures of the CEOs from each year for age, sex, race, dominant emotion, and overall emotion between happy, angry, disgust, fear, surpise, and neutral.")
    st.write("Additioanlly, we reviewed daily firm returns and yearly firm returns of each CEO during their tenure. This gave us a better understanding of how the company performed under the CEO and allowed us to evaluate patterns between firm performance and CEO attributes.")
    

elif page == "CEO Attributes":
    st.title("🧠 CEO Attributes and Firm Returns: CEO Attributes")

    # Load Returns Data
    @st.cache_data
    def load_returns_data():
        returns_file = "outputs/output_yearly_with_image_paths.csv"
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

    # Load CEO Data
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

    # Dropdowns for company and year selection
    st.subheader("View attributes by Company")
    all_tickers = sorted(ceo_df['Ticker'].unique())
    selected_company = st.selectbox("Please select company", all_tickers)
    allowed_years = list(range(2010, 2020))
    selected_year = st.selectbox("Please select year", allowed_years)

    # Filter data
    filtered_data = ceo_df[
        (ceo_df['Ticker'] == selected_company) &
        (ceo_df['Year'] == selected_year)
    ]

    if filtered_data.empty:
        st.warning("No data available for this company and year.")
        selected_data = None
    else:
        selected_data = filtered_data.iloc[0]

        # Layout: image | key info
        img_col, info_col = st.columns([1, 2])

        with img_col:
            image_path = selected_data.get('Image Path')
            if pd.notna(image_path) and os.path.exists(image_path):
                st.image(image_path, width=200)
            else:
                st.info("Image not available for this CEO/year.")

        with info_col:
            st.markdown(f"**Company:** {selected_data.get('Ticker', 'N/A')}")
            st.markdown(f"**Year:** {selected_data.get('Year', 'N/A')}")
            st.markdown(f"**CEO:** {selected_data.get('CEO', 'N/A')}")
            firm_return = selected_data.get('Tenure_Cum_Ret_Overall')
            if pd.notna(firm_return):
                st.metric(label="Firm Return", value=f"{firm_return:.1f}%")
            else:
                st.metric(label="Firm Return", value="N/A")

        # Attribute table below
        st.markdown("---")
        st.subheader("CEO Attribute Details")

        ceo_attributes = {
            "Sex": selected_data.get('dominant_gender', 'N/A'),
            "Race (Inferred)": selected_data.get('dominant_race', 'N/A'),
            "Age": selected_data.get('age', 'N/A'),
            "Dominant Emotion": selected_data.get('dominant_emotion', 'N/A'),
            "Angry": f"{selected_data.get('angry', 0):.2f}",
            "Disgust": f"{selected_data.get('disgust', 0):.2f}",
            "Fear": f"{selected_data.get('fear', 0):.2f}",
            "Happy": f"{selected_data.get('happy', 0):.2f}",
            "Sad": f"{selected_data.get('sad', 0):.2f}",
            "Surprise": f"{selected_data.get('surprise', 0):.2f}",
            "Neutral": f"{selected_data.get('neutral', 0):.2f}",
        }
        attr_df = pd.DataFrame(list(ceo_attributes.items()), columns=["Attribute", "Value"])
        st.dataframe(attr_df.reset_index(drop=True))

    # Load daily returns
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

    # Plot cumulative returns
    if 'selected_data' in locals() and selected_data is not None:
        ceo_name = selected_data.get('CEO')
    else:
        st.error("selected_data is not defined or is None.")
        st.stop()

    if 'selected_company' not in locals():
        st.error("selected_company is not defined.")
        st.stop()

    def plot_cumulative_returns_by_ceo(ticker, ceo_name, r_df):
        r_df['Date'] = pd.to_datetime(r_df['Date'], errors='coerce')

        ceo_data = r_df[(r_df['Ticker'] == ticker) & (r_df['CEO'] == ceo_name)].copy()

        if ceo_data.empty:
            st.warning("No data available for the selected CEO and company.")
            return None

        mask_dates = (r_df['Date'] >= '2010-01-01') & (r_df['Date'] <= '2019-12-31')
        r_df = r_df[mask_dates]
        r_df = r_df.set_index('Date')

        firm_df = r_df[(r_df['Ticker'] == ticker) & (r_df['CEO'] == ceo_name)].copy()
        spy_df = r_df[(r_df['Ticker'] == 'SPY') & (r_df.index.isin(firm_df.index))].copy()

        if firm_df.empty:
            st.warning("Company return data not available for CEO's tenure.")
            return None

        firm_df = firm_df.sort_index()
        firm_df['Return'] = firm_df['Return'].astype(float)
        cum_firm = (1 + firm_df['Return']).cumprod() - 1

        plot_spy = not spy_df.empty
        if plot_spy:
            spy_df = spy_df.sort_index()
            spy_df['Return'] = spy_df['Return'].astype(float)
            cum_spy = (1 + spy_df['Return']).cumprod() - 1

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=cum_firm.index, y=cum_firm, ax=ax, label=ticker)
        if plot_spy:
            sns.lineplot(x=cum_spy.index, y=cum_spy, ax=ax, label='SPY')

        plt.title(f"Daily Returns for {ceo_name} at {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    fig = plot_cumulative_returns_by_ceo(selected_company, ceo_name, r_df)
    if fig:
        st.pyplot(fig)


elif page == "Analysis":
    st.title("📈 CEO Attributes and Firm Returns:Analysis")
    st.write("Visualize returns and evaluate CEO performance.")
