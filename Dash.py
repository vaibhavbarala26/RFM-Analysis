# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Caching Data Loading and Processing ---
@st.cache_data
def load_and_process_data():
    # Load and combine your datasets
    try:
        df1 = pd.read_csv("./online_retail_II (2).csv", encoding="latin1")
        df2 = pd.read_csv("./online_retail_II.csv", encoding="latin1")
        df = pd.concat([df1, df2])
    except FileNotFoundError:
        st.error("Error: Make sure 'online_retail_II (2).csv' and 'online_retail_II.csv' are in the same folder.")
        return None, None

    # --- RFM Data Preparation ---
    df.dropna(subset=['Customer ID'], inplace=True)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M")

    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm_df.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)

    r_labels, f_labels, m_labels = range(4, 0, -1), range(1, 5), range(1, 5)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=4, labels=r_labels, duplicates='drop').astype(int)
    rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=[0, 1, 3, 7, rfm_df['Frequency'].max()], labels=f_labels).astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['MonetaryValue'], q=4, labels=m_labels, duplicates='drop').astype(int)
    rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

    def assign_segment(score):
        if score >= 11: return 'Champions'
        elif score >= 9: return 'Loyal Customers'
        elif score >= 6: return 'Potential Loyalists'
        elif score >= 5: return 'At-Risk Customers'
        else: return "Can't Lose Them"
    rfm_df['Segment'] = rfm_df['RFM_Score'].apply(assign_segment)

    country_df = df[['Customer ID', 'Country']].drop_duplicates(subset=['Customer ID'])
    rfm_with_country = rfm_df.merge(country_df, on='Customer ID', how='left')

    # --- AI Prediction Model Preparation ---
    cltv_df = df.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda date: (date.max() - date.min()).days, lambda date: (snapshot_date - date.min()).days],
         'Invoice': 'nunique', 'TotalPrice': 'sum'}
    )
    cltv_df.columns = ['Recency_CLTV', 'T', 'Frequency_CLTV', 'Monetary_CLTV']
    cltv_df = cltv_df[cltv_df['Monetary_CLTV'] > 0]
    cltv_df['Frequency_CLTV'] /= cltv_df['T']
    cltv_df['Recency_CLTV'] /= cltv_df['T']
    cltv_df = cltv_df[cltv_df['Frequency_CLTV'] > 0]

    X = cltv_df.drop('T', axis=1)
    y = cltv_df['T']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  verbose=False)
    
    predictions = xgb_model.predict(X)
    cltv_df['Predicted_Next_Purchase_Day'] = predictions.astype(int)
    prediction_results = cltv_df[['Predicted_Next_Purchase_Day']].reset_index()

    return rfm_with_country, prediction_results

# Load the data using the cached function
rfm_data, prediction_data = load_and_process_data()

# --- Dashboard Title ---
st.title("📊 Customer Analytics Dashboard")
st.markdown("---")

# --- Sidebar for Filters ---
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox(
    "Select Country",
    options=sorted(rfm_data['Country'].unique()),
    index=list(sorted(rfm_data['Country'].unique())).index("United Kingdom") # Default to UK
)

# Filter data based on selection
filtered_rfm = rfm_data[rfm_data['Country'] == selected_country]

# --- Main Page Layout ---

# Create tabs
tab1, tab2 = st.tabs(["RFM Segmentation Analysis", "Next Purchase Day Prediction"])

with tab1:
    st.header(f"RFM Analysis for {selected_country}")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{filtered_rfm.shape[0]:,}")
    col2.metric("Avg. Recency (Days)", f"{filtered_rfm['Recency'].mean():.1f}")
    col3.metric("Avg. Frequency", f"{filtered_rfm['Frequency'].mean():.1f}")
    col4.metric("Avg. Monetary Value ($)", f"{filtered_rfm['MonetaryValue'].mean():.2f}")

    st.markdown("---")

    # Charts
    c1, c2 = st.columns((7, 5))
    with c1:
        st.subheader("Customer Segments Distribution")
        segment_counts = filtered_rfm['Segment'].value_counts()
        fig_bar = px.bar(
            segment_counts, x=segment_counts.index, y=segment_counts.values,
            color=segment_counts.index, labels={'y': 'Number of Customers', 'x': 'Segment'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Recency vs. Frequency")
        fig_scatter = px.scatter(
            filtered_rfm, x='Recency', y='Frequency', color='Segment',
            hover_data=['MonetaryValue']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.header("Next Purchase Day Prediction")
    
    c1, c2 = st.columns((4, 8))
    with c1:
        st.subheader("Marketing Actions")
        st.success("**Immediate Action (0-5 Days):** Target with 'New Arrivals' or 'Top Sellers' campaigns.")
        st.info("**Nurture (15-45 Days):** Schedule automated marketing just before their predicted purchase date.")
        st.error("**Win-Back (> 90 Days):** Use special win-back campaigns and offers to re-engage.")

    with c2:
        st.subheader("Predicted Days for Each Customer")
        st.dataframe(
            prediction_data,
            use_container_width=True,
            height=500
        )
