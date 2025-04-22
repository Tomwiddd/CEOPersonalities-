import streamlit as st
import pandas as pd
import numpy as np

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

# 2. CEO/Company Data (Example with 2 companies over 2 years)
# Replace 'placeholder_image_url_1.jpg', etc. with actual image paths or URLs
# For local files use the path, for web images use the URL.
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
    'Image URL': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Tim_Cook_2019.jpg/440px-Tim_Cook_2019.jpg', # Example URL for Tim Cook
        'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Tim_Cook_2019.jpg/440px-Tim_Cook_2019.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/440px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg', # Example URL for Mark Zuckerberg
        'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/440px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg'
    ]
}
ceo_df = pd.DataFrame(ceo_data)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide") # Use wide layout

st.title("Study of CEO Headshot Attributes and Firm Returns")

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


# --- How to Run ---
# 1. Save the code above as a Python file (e.g., `ceo_app.py`).
# 2. Make sure you have streamlit and pandas installed:
#    pip install streamlit pandas
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the app using:
#    streamlit run ceo_app.py