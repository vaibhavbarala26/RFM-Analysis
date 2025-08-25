# app.py

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc # For better styling

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb

# --- 1. DATA PROCESSING (From your notebook) ---

# Load and combine your datasets
try:
    df1 = pd.read_csv("./online_retail_II (2).csv", encoding="latin1")
    df2 = pd.read_csv("./online_retail_II.csv", encoding="latin1")
    df = pd.concat([df1, df2])
except FileNotFoundError:
    print("Error: Make sure 'online_retail_II (2).csv' and 'online_retail_II.csv' are in the same folder.")
    exit()

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
    {
        'InvoiceDate': [lambda date: (date.max() - date.min()).days, lambda date: (snapshot_date - date.min()).days],
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }
)
cltv_df.columns = ['Recency_CLTV', 'T', 'Frequency_CLTV', 'Monetary_CLTV']
cltv_df = cltv_df[cltv_df['Monetary_CLTV'] > 0]
cltv_df['Frequency_CLTV'] = cltv_df['Frequency_CLTV'] / cltv_df['T']
cltv_df['Recency_CLTV'] = cltv_df['Recency_CLTV'] / cltv_df['T']
cltv_df = cltv_df[(cltv_df['Frequency_CLTV'] > 0)]

# Train XGBoost Model
X = cltv_df.drop('T', axis=1)
y = cltv_df['T']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  verbose=False)

# Get predictions
predictions = xgb_model.predict(X)
cltv_df['Predicted_Next_Purchase_Day'] = predictions.astype(int)
prediction_results = cltv_df[['Predicted_Next_Purchase_Day']].reset_index()


# --- 2. DASH APP LAYOUT ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    # Header
    dbc.Row(
        dbc.Col(html.H1("Customer Analytics Dashboard", className="text-center text-primary, mb-4"), width=12)
    ),

    # Tabs for different sections
    dbc.Tabs([
        # Tab 1: RFM Segmentation
        dbc.Tab(label="RFM Segmentation Analysis", children=[
            # Filters
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in sorted(rfm_with_country['Country'].unique())],
                        value='United Kingdom'
                    ), width=6
                )
            ], className="mb-4 mt-4"),

            # KPI Cards
            dbc.Row([
                dbc.Col(dbc.Card(id='total-customers-card', body=True, color="primary", inverse=True), width=3),
                dbc.Col(dbc.Card(id='avg-recency-card', body=True, color="success", inverse=True), width=3),
                dbc.Col(dbc.Card(id='avg-frequency-card', body=True, color="info", inverse=True), width=3),
                dbc.Col(dbc.Card(id='avg-monetary-card', body=True, color="warning", inverse=True), width=3),
            ], className="mb-4"),

            # Main Plots
            dbc.Row([
                dbc.Col(dcc.Graph(id='segment-bar-chart'), width=7),
                dbc.Col(dcc.Graph(id='rfm-scatter-plot'), width=5),
            ]),
        ]),

        # Tab 2: AI Predictions
        dbc.Tab(label="Next Purchase Day Prediction", children=[
            dbc.Row([
                # Explanation Column
                dbc.Col([
                    html.H4("Marketing Actions Based on Predictions", className="mt-4"),
                    html.Hr(),
                    dbc.Card([
                        dbc.CardHeader("Immediate Action (0-5 Days)"),
                        dbc.CardBody(
                            "These are your most loyal customers. Target them with 'New Arrivals' or 'Top Sellers' campaigns. Avoid 'we miss you' discounts."
                        )
                    ], color="success", outline=True, className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Nurture (15-45 Days)"),
                        dbc.CardBody(
                            "These are reliable, regular customers. Schedule automated marketing emails to trigger just before their predicted purchase date."
                        )
                    ], color="info", outline=True, className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Win-Back (> 90 Days)"),
                        dbc.CardBody(
                            "These customers are at risk of lapsing. Use special win-back campaigns, offers, or surveys to re-engage them."
                        )
                    ], color="danger", outline=True),
                ], width=4),

                # Data Table Column
                dbc.Col([
                    html.H4("Predicted Next Purchase Day for Each Customer", className="mt-4"),
                    dash_table.DataTable(
                        id='prediction-table',
                        columns=[{"name": i, "id": i} for i in prediction_results.columns],
                        data=prediction_results.to_dict('records'),
                        page_size=15,
                        sort_action="native",
                        filter_action="native",
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                    )
                ], width=8),
            ])
        ])
    ])
], fluid=True)


# --- 3. CALLBACKS TO MAKE THE DASHBOARD INTERACTIVE ---

# Callback for RFM Tab
@app.callback(
    [
        Output('segment-bar-chart', 'figure'),
        Output('rfm-scatter-plot', 'figure'),
        Output('total-customers-card', 'children'),
        Output('avg-recency-card', 'children'),
        Output('avg-frequency-card', 'children'),
        Output('avg-monetary-card', 'children'),
    ],
    [Input('country-dropdown', 'value')]
)
def update_rfm_analysis(selected_country):
    if not selected_country:
        raise dash.exceptions.PreventUpdate

    filtered_rfm = rfm_with_country[rfm_with_country['Country'] == selected_country]

    # Bar Chart
    segment_counts = filtered_rfm['Segment'].value_counts()
    bar_fig = px.bar(
        segment_counts, x=segment_counts.index, y=segment_counts.values,
        title=f'Customer Segments in {selected_country}', color=segment_counts.index,
        labels={'y': 'Number of Customers', 'x': 'Segment'}
    )

    # Scatter Plot
    scatter_fig = px.scatter(
        filtered_rfm, x='Recency', y='Frequency', color='Segment',
        title='Recency vs. Frequency by Segment', hover_data=['MonetaryValue']
    )

    # KPI Cards
    total_customers = f"Total Customers: {filtered_rfm.shape[0]}"
    avg_recency = f"Avg. Recency: {filtered_rfm['Recency'].mean():.1f} days"
    avg_frequency = f"Avg. Frequency: {filtered_rfm['Frequency'].mean():.1f}"
    avg_monetary = f"Avg. Monetary Value: ${filtered_rfm['MonetaryValue'].mean():.2f}"

    return bar_fig, scatter_fig, total_customers, avg_recency, avg_frequency, avg_monetary

# No callback needed for the prediction tab as it's pre-calculated and static.

# --- 4. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)