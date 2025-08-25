# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Caching Data Loading ---
# Load the preprocessed data from the Parquet file
# This is much faster than loading and processing CSVs
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("customer_data.parquet")
        return df
    except FileNotFoundError:
        st.error("Error: The 'customer_data.parquet' file was not found. Please run the 'preprocess.py' script first.")
        return None

# Load the data
data = load_data()

if data is not None:
    # --- Dashboard Title ---
    st.title("📊 Customer Analytics Dashboard")
    st.markdown("---")

    # --- Sidebar for Filters ---
    st.sidebar.header("Filters")
    selected_country = st.sidebar.selectbox(
        "Select Country",
        options=sorted(data['Country'].dropna().unique()),
        index=list(sorted(data['Country'].dropna().unique())).index("United Kingdom")
    )

    # Filter data based on selection
    filtered_data = data[data['Country'] == selected_country]

    # --- Main Page Layout ---
    tab1, tab2 = st.tabs(["RFM Segmentation Analysis", "Next Purchase Day Prediction"])

    with tab1:
        st.header(f"RFM Analysis for {selected_country}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{filtered_data.shape[0]:,}")
        col2.metric("Avg. Recency (Days)", f"{filtered_data['Recency'].mean():.1f}")
        col3.metric("Avg. Frequency", f"{filtered_data['Frequency'].mean():.1f}")
        col4.metric("Avg. Monetary Value ($)", f"{filtered_data['MonetaryValue'].mean():.2f}")
        st.markdown("---")
        c1, c2 = st.columns((7, 5))
        with c1:
            st.subheader("Customer Segments Distribution")
            segment_counts = filtered_data['Segment'].value_counts()
            fig_bar = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values,
                             color=segment_counts.index, labels={'y': 'Number of Customers', 'x': 'Segment'})
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            st.subheader("Recency vs. Frequency")
            fig_scatter = px.scatter(filtered_data, x='Recency', y='Frequency', color='Segment',
                                     hover_data=['MonetaryValue'])
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.header("Next Purchase Day Prediction")
        c1, c2 = st.columns((4, 8))
        with c1:
            st.subheader("Marketing Actions")
            st.success("**Immediate Action (0-5 Days):** Target with 'New Arrivals'.")
            st.info("**Nurture (15-45 Days):** Schedule automated marketing.")
            st.error("**Win-Back (> 90 Days):** Use win-back campaigns.")
        with c2:
            st.subheader("Predicted Days for Each Customer")
            prediction_display = filtered_data[['Customer ID', 'Predicted_Next_Purchase_Day']].dropna()
            st.dataframe(prediction_display, use_container_width=True, height=500)
