import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- Data Loading ---

# Load monthly returns data (assuming the file exists)
@st.cache_data
def load_returns_data():
    # Update this path to where your CSV file is located
    returns_file = "monthly_returns_2010_2025.csv"
    if os.path.exists(returns_file):
        # Assuming CSV has columns like Date, AAPL, META, SPY, etc.
        df = pd.read_csv(returns_file)
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    else:
        # If file doesn't exist, create dummy data for demonstration
        date_range = pd.date_range(start='2010-01-01', end='2025-12-31', freq='M')
        tickers = ['AAPL', 'META', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
        
        # Create random returns with some correlation
        np.random.seed(42)  # For reproducibility
        n_dates = len(date_range)
        n_tickers = len(tickers)
        
        # Base market return (SPY like)
        market_returns = np.random.normal(0.008, 0.04, n_dates)  # Monthly mean ~1%, SD ~4%
        
        all_returns = {}
        for i, ticker in enumerate(tickers):
            if ticker == 'SPY':
                all_returns[ticker] = market_returns
            else:
                # Add some correlation to market with company-specific noise
                beta = np.random.uniform(0.8, 1.5)  # Random beta
                stock_returns = beta * market_returns + np.random.normal(0.002, 0.08, n_dates)
                all_returns[ticker] = stock_returns
                
        df = pd.DataFrame(all_returns, index=date_range)
        return df

# --- Pseuso Data Creation ---

# 1. Correlation Data
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

# Function to apply color based on value for the correlation table
def color_correlation(val):
    if val > 0.05:
        color = 'red'
    elif val > 0.02:
        color = 'lightcoral' # Adjusted for visibility based on screenshot
    elif val < -0.05:
        color = 'blue'
    elif val < -0.01:
        color = 'lightblue' # Adjusted for visibility
    else:
        color = 'black'
    return f'color: {color}'

# 2. CEO/Company Data (Extended with tenure information)
ceo_data = {
    'Company': ['AAPL', 'AAPL', 'META', 'META'],
    'Year': [2019, 2020, 2019, 2020],
    'CEO': ['Tim Cook', 'Tim Cook', 'Mark Zuckerberg', 'Mark Zuckerberg'],
    'Sex': ['M', 'M', 'M', 'M'],
    'Race': ['W', 'W', 'W', 'W'], # Using 'W' as placeholder based on image
    'Age': [58, 59, 35, 36], # Example ages
    'Angry': [0.01, 0.02, 0.03, 0.01], # Random sentiment scores
    'Disgust': [0.05, 0.04, 0.06, 0.07],
    'Fear': [0.00, 0.01, 0.00, 0.02],
    'Happy': [0.10, 0.15, 0.08, 0.12],
    'Sad': [0.02, 0.01, 0.03, 0.02],
    'Surprise': [0.00, 0.00, 0.01, 0.00],
    'Neutral': [0.70, 0.65, 0.75, 0.70],
    'Attractiveness': [0.7, 0.72, 0.65, 0.66],
    'Firm Return': [0.1, 0.15, 0.08, 0.12], # Example firm returns
    'Tenure Start': ['2011-08-24', '2011-08-24', '2004-02-04', '2004-02-04'],  # Adding CEO tenure start
    'Image URL': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Visit_of_Tim_Cook_to_the_European_Commission_-_P061904-946789.jpg/1024px-Visit_of_Tim_Cook_to_the_European_Commission_-_P061904-946789.jpg', # Example URL for Tim Cook
        'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Visit_of_Tim_Cook_to_the_European_Commission_-_P061904-946789.jpg/1024px-Visit_of_Tim_Cook_to_the_European_Commission_-_P061904-946789.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/440px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg', # Example URL for Mark Zuckerberg
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/440px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg'
    ]
}
ceo_df = pd.DataFrame(ceo_data)
ceo_df['Tenure Start'] = pd.to_datetime(ceo_df['Tenure Start'])

# Function to plot cumulative returns
def plot_cumulative_returns(ticker, start_date, returns_df):
    # Create a copy to avoid modifying the original
    plot_df = returns_df.copy()
    
    # Filter data from start_date onwards
    if start_date in plot_df.index:
        plot_df = plot_df[plot_df.index >= start_date]
    else:
        # Find the next available date if start_date is not in the index
        plot_df = plot_df[plot_df.index >= plot_df.index[plot_df.index > start_date][0]]
    
    # Check if we have data for this ticker
    if ticker not in plot_df.columns or 'SPY' not in plot_df.columns:
        st.error(f"Return data not available for {ticker} or SPY")
        return None
    
    # Calculate cumulative returns (adding 1 to each return and using cumprod)
    plot_df['Cum_Return_' + ticker] = (1 + plot_df[ticker]).cumprod() - 1
    plot_df['Cum_Return_SPY'] = (1 + plot_df['SPY']).cumprod() - 1
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x=plot_df.index, y='Cum_Return_' + ticker, ax=ax, label=ticker)
    sns.lineplot(data=plot_df, x=plot_df.index, y='Cum_Return_SPY', ax=ax, label='SPY')
    
    plt.title(f"Cumulative Returns: {ticker} vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

# --- Streamlit App Layout ---

st.set_page_config(layout="wide") # Use wide layout

st.title("Study of CEO Headshot Attributes and Firm Returns")

# Load returns data
returns_df = load_returns_data()

col1, col2 = st.columns([1, 2]) # Define columns for layout (adjust ratio as needed)

# --- Column 1: Correlation Table ---
with col1:
    st.subheader("Correlation between image attributes and firm return")
    # Apply styling to the correlation DataFrame
    st.dataframe(correlation_df.style.applymap(color_correlation, subset=['Correlation (ret_0)'])
                 .format({'Correlation (ret_0)': "{:.3f}"})) # Format numbers to 3 decimal places

# --- Column 2: View Attributes by Company ---
with col2:
    st.subheader("View attributes by Company")

    # Dropdowns for selecting company and year
    available_companies = sorted(ceo_df['Company'].unique())
    selected_company = st.selectbox("Please select company", available_companies)

    available_years = sorted(ceo_df[ceo_df['Company'] == selected_company]['Year'].unique())
    selected_year = st.selectbox("Please select year", available_years)

    # Filter data based on selections
    selected_data = ceo_df[(ceo_df['Company'] == selected_company) & (ceo_df['Year'] == selected_year)].iloc[0]

    # Display CEO Info and Image
    info_col, img_col = st.columns([1, 1]) # Sub-columns within col2

    with img_col:
        if pd.notna(selected_data['Image URL']):
             st.image(selected_data['Image URL'], width=200) # Adjust width as needed
        else:
             st.write("Image not available") # Placeholder if no image URL
        st.metric(label="Firm Return", value=f"{selected_data['Firm Return']:.1f}") # Display Firm Return below image

    with info_col:
        st.text(f"Company: {selected_data['Company']}")
        st.text(f"Year: {selected_data['Year']}")
        st.text(f"CEO: {selected_data['CEO']}")
        st.text(f"Sex: {selected_data['Sex']}")
        # Add hedging for inferred attributes like Race
        st.text(f"Race (inferred): {selected_data['Race']}")
        st.text(f"Age: {selected_data['Age']}")
        st.text(f"Angry: {selected_data['Angry']:.2f}")
        st.text(f"Disgust: {selected_data['Disgust']:.2f}")
        st.text(f"Fear: {selected_data['Fear']:.2f}")
        st.text(f"Happy: {selected_data['Happy']:.2f}")
        st.text(f"Sad: {selected_data['Sad']:.2f}")
        st.text(f"Surprise: {selected_data['Surprise']:.2f}")
        st.text(f"Neutral: {selected_data['Neutral']:.2f}")
        st.text(f"Attractiveness: {selected_data['Attractiveness']:.1f}")

# New section for cumulative returns plot
st.subheader(f"Cumulative Returns During {selected_data['CEO']}'s Tenure")

# Get the tenure start date for the selected company/CEO
tenure_start = selected_data['Tenure Start']

# Create and display the plot
if selected_company in returns_df.columns and 'SPY' in returns_df.columns:
    fig = plot_cumulative_returns(selected_company, tenure_start, returns_df)
    if fig:
        st.pyplot(fig)
else:
    st.error(f"Return data not available for {selected_company} or SPY. Please check your data file.")

# --- How to Run ---
# 1. Save the code above as a Python file (e.g., `ceo_app.py`).
# 2. Make sure you have streamlit, pandas, matplotlib and seaborn installed:
#    pip install streamlit pandas matplotlib seaborn
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the app using:
#    streamlit run ceo_app.py