import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

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
    output_file = "outputs/output_yearly_with_image_paths.csv"
    output_yearly = pd.read_csv(output_file)
    output_yearly.columns = output_yearly.columns.str.strip()

    
    st.title("📈 CEO Attributes and Firm Returns: Analysis")
    
    st.subheader("CEO Race Representation (2010–2019)")
    
    # Define race columns
    race_cols = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
    
    # Melt the race data into long format
    df_race = output_yearly[race_cols].melt(var_name='Race', value_name='Percentage')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_race, x='Race', y='Percentage', estimator='mean', ci='sd', ax=ax)
    ax.set_title('CEO Race Representation')
    ax.set_ylabel('Mean Percentage (%)')
    ax.set_xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)

    st.subheader("CEO Sex Representation (2010–2019)")
    
    # Define sex columns
    sex_cols = ['Man', 'Woman']
    
    # Melt the sex data into long format
    df_sex = output_yearly[sex_cols].melt(var_name='Sex', value_name='Percentage')
    
    # Create the plot
    fig_sex, ax_sex = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_sex, x='Sex', y='Percentage', estimator='mean', ci='sd', ax=ax_sex)
    ax_sex.set_title('CEO Sex Representation')
    ax_sex.set_ylabel('Mean Percentage (%)')
    ax_sex.set_xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig_sex)

    st.subheader("CEO Age Distribution (2010–2019)")

    # Create the age distribution plot
    fig_age, ax_age = plt.subplots(figsize=(10, 6))
    sns.histplot(output_yearly['Age'], bins=20, kde=True, ax=ax_age)
    ax_age.set_title('Age Distribution')
    ax_age.set_xlabel('Age')
    ax_age.set_ylabel('Count')
    ax_age.grid(True)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig_age)

    st.subheader("📌 Correlation Between CEO Attributes and Firm Returns")
    
    # Define relevant columns
    image_attributes = [
        'Age', 'Woman', 'Man', 'asian', 'indian', 'black', 'white',
        'middle eastern', 'latino hispanic', 'angry', 'disgust', 'fear',
        'happy', 'sad', 'surprise', 'neutral'
    ]
    return_measures = ['Year_Cum_Ret_Overall', 'Tenure_Cum_Ret_Overall']
    
    # Calculate correlation matrix
    correlation_matrix = output_yearly[image_attributes + return_measures].corr()
    
    # Extract correlation values between attributes and return measures
    correlation_table = correlation_matrix.loc[image_attributes, return_measures]
    
    # Apply formatting for visual clarity
    styled_corr = correlation_table.style.background_gradient(cmap='coolwarm').format("{:.3f}")
    
    # Display in Streamlit
    st.dataframe(styled_corr, use_container_width=True)

    # # Load your data (you might need to use st.cache or st.file_uploader in real apps)
    # df = output_daily.copy()
    
    # # Ensure 'Date' column is datetime
    # if 'Date' in df.columns:
    #     df['Date'] = pd.to_datetime(df['Date'])
    
    # # Drop rows with missing required values
    # df = df.dropna(subset=['CEO', 'Year', 'Year_Cum_Ret_Daily'])
    
    # # Set attributes
    # image_attributes = ['Age', 'Woman', 'Man', 'asian', 'indian', 'black', 'white', 
    #                     'middle eastern', 'latino hispanic', 'angry', 'disgust', 
    #                     'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # # Create CEO-Year selector
    # df['CEO_Year'] = df['CEO'] + " (" + df['Year'].astype(str) + ")"
    # selected = st.selectbox("Select CEO-Year", df['CEO_Year'].unique())
    
    # # Filter based on selection
    # selected_ceo, selected_year = selected.rsplit(" (", 1)
    # selected_year = int(selected_year.rstrip(")"))
    # subset = df[(df['CEO'] == selected_ceo) & (df['Year'] == selected_year)]
    
    # # Plot line chart
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=subset['Date'],
    #     y=subset['Year_Cum_Ret_Daily'],
    #     mode='lines',
    #     name=f'{selected_ceo} ({selected_year})'
    # ))
    
    # fig.update_layout(
    #     title=f'Daily Cumulative Return: {selected_ceo} ({selected_year})',
    #     xaxis_title='Date',
    #     yaxis_title='Year_Cum_Ret_Daily',
    #     height=500
    # )
    
    # st.plotly_chart(fig)
    
    # # Show attribute table
    # st.subheader("CEO Image Attributes")
    # ceo_row = subset.iloc[0]
    # attr_df = pd.DataFrame({
    #     "Attribute": image_attributes,
    #     "Value": [round(ceo_row[attr], 3) if pd.notnull(ceo_row[attr]) else 'N/A' for attr in image_attributes]
    # })
    # st.table(attr_df)
