import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import base64
import os
import warnings
warnings.filterwarnings('ignore')

from data_processing import (
    load_and_clean_data, filter_data, get_executive_kpis,
    get_monthly_sales_trend, get_category_revenue, get_sub_category_revenue,
    get_product_revenue, get_top_products, get_sales_distribution,
    get_region_performance, get_profit_analysis,
    get_inventory_status, get_stock_aging, get_inventory_alerts,
    get_forecast, get_customer_segments, get_customer_recommendations,
    get_business_insights, get_seasonal_patterns, get_ai_response, get_ceo_report
)

from ml_engine import (
    train_sales_models, predict_sales_single,
    train_demand_forecast_model, analyze_demand_trends,
    train_customer_segmentation, calculate_inventory_optimization
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & FAST STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail & Inventory ML Manager",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors
DARK_BG = "#0b0f19"
DARK_CARD = "#151b2b"
DARK_TEXT = "#f1f5f9"
DARK_TEXT2 = "#94a3b8"
LIGHT_BG = "#f8fafc"
LIGHT_CARD = "#ffffff"
LIGHT_TEXT = "#0f172a"
LIGHT_TEXT2 = "#64748b"
ACCENT = "#4f46e5"     # Indigo
ACCENT2 = "#06b6d4"    # Cyan
ACCENT3 = "#ec4899"    # Pink

if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

dark_mode = st.session_state.theme == 'Dark'
bg = DARK_BG if dark_mode else LIGHT_BG
card = DARK_CARD if dark_mode else LIGHT_CARD
text = DARK_TEXT if dark_mode else LIGHT_TEXT
text2 = DARK_TEXT2 if dark_mode else LIGHT_TEXT2
border = 'rgba(255,255,255,0.08)' if dark_mode else 'rgba(0,0,0,0.06)'

# Clean, lightweight CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {bg}; color: {text}; }}

    /* KPI Cards */
    .kpi-card {{
        background: {card};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border: 1px solid {border};
        box-shadow: {'0 4px 14px rgba(0,0,0,0.25)' if dark_mode else '0 2px 6px rgba(0,0,0,0.04)'};
        position: relative;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text2};
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {text};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }}
    .growth-pos {{ color: #10b981; }}
    .growth-neg {{ color: #ef4444; }}

    /* Page Titles */
    .page-title {{
        font-size: 1.7rem;
        font-weight: 800;
        color: {text};
        margin-bottom: 0.2rem;
    }}
    .page-subtitle {{
        font-size: 0.85rem;
        color: {text2};
        margin-bottom: 1.2rem;
    }}

    /* Friendly Simple English Explanation Box */
    .explainer-box {{
        background: {'rgba(79, 70, 229, 0.12)' if dark_mode else '#eef2ff'};
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: {text};
    }}
    .explainer-title {{
        font-weight: 700;
        font-size: 0.95rem;
        color: {ACCENT if not dark_mode else '#a5b4fc'};
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* Action Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, #4338ca);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.45rem 1.2rem;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #4338ca, #3730a3);
        transform: translateY(-1px);
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HIGH-SPEED CACHED DATA & ML LOADERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data():
    return load_and_clean_data('data/superstore.csv')

df_raw = get_clean_data()

@st.cache_resource(show_spinner=False)
def get_sales_models():
    return train_sales_models(df_raw)

@st.cache_resource(show_spinner=False)
def get_forecast_models():
    return train_demand_forecast_model(df_raw, freq='M')

@st.cache_resource(show_spinner=False)
def get_kmeans_models():
    return train_customer_segmentation(df_raw)


# -----------------------------------------------------------------------------
# HELPER RENDERING FUNCTIONS
# -----------------------------------------------------------------------------
def render_kpi(label, value, growth=None, prefix="", suffix=""):
    growth_html = ""
    if growth is not None:
        pos = growth >= 0
        arrow = "↑" if pos else "↓"
        cls = "growth-pos" if pos else "growth-neg"
        growth_html = f'<div class="kpi-sub"><span class="{cls}">{arrow} {abs(growth):.1f}%</span> vs prev period</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{prefix}{value}{suffix}</div>
        {growth_html}
    </div>
    """, unsafe_allow_html=True)

def render_explainer(title, purpose, what_it_shows):
    """Renders a simple, clean English explanation box below every dashboard."""
    st.markdown(f"""
    <div class="explainer-box">
        <div class="explainer-title">📌 {title}</div>
        <div style="margin-bottom: 0.35rem;"><strong>Dashboard Purpose:</strong> {purpose}</div>
        <div><strong>What It Shows:</strong> {what_it_shows}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.6rem 0 0.4rem 0;">
        <div style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, {ACCENT}, {ACCENT3}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            RetailPro AI
        </div>
        <div style="font-size: 0.72rem; color: {text2}; font-weight: 500;">
            Retail & Inventory Management System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin: 0.5rem 0; opacity: 0.2;">', unsafe_allow_html=True)
    st.caption("SELECT SECTION")

    pages = [
        "📊 Executive Dashboard",
        "📈 Sales Analytics",
        "🤖 ML Sales Prediction",
        "🔮 Demand Forecasting",
        "👥 Customer Segmentation",
        "📦 Inventory Optimization",
        "🏆 Product Performance",
        "🔬 ML Model Evaluation",
        "💬 AI Retail Assistant"
    ]

    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown('<hr style="margin: 0.6rem 0; opacity: 0.2;">', unsafe_allow_html=True)
    st.caption("QUICK FILTERS")

    with st.expander("Filter Data", expanded=False):
        cat_list = ['All'] + sorted(df_raw['Category'].unique().tolist())
        selected_cat = st.selectbox("Category", cat_list)
        reg_list = ['All'] + sorted(df_raw['Region'].unique().tolist())
        selected_reg = st.selectbox("Region", reg_list)

    st.markdown('<hr style="margin: 0.6rem 0; opacity: 0.2;">', unsafe_allow_html=True)
    st.caption("THEME")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️ Light", use_container_width=True, disabled=(st.session_state.theme == 'Light')):
            st.session_state.theme = 'Light'
            st.rerun()
    with c2:
        if st.button("🌙 Dark", use_container_width=True, disabled=(st.session_state.theme == 'Dark')):
            st.session_state.theme = 'Dark'
            st.rerun()

    st.markdown(f"""
    <div style="font-size: 0.65rem; color: {text2}; text-align: center; margin-top: 1.5rem;">
        Dataset: Superstore (9,994 Orders)<br>
        B.Tech CSE Project Edition
    </div>
    """, unsafe_allow_html=True)

# Apply global quick filters
cat_filter = [] if selected_cat == 'All' else [selected_cat]
reg_filter = [] if selected_reg == 'All' else [selected_reg]
filtered_df = filter_data(df_raw, None, cat_filter, [], reg_filter, [])


# =============================================================================
# 1. EXECUTIVE DASHBOARD
# =============================================================================
if selected_page == "📊 Executive Dashboard":
    st.markdown('<div class="page-title">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-level summary of store sales, profit, orders, and business health.</div>', unsafe_allow_html=True)

    kpis = get_executive_kpis(filtered_df)
    rev, rev_g = kpis['Total Revenue']
    prof, prof_g = kpis['Total Profit']
    orders, ord_g = kpis['Total Orders']
    sold, _ = kpis['Total Products Sold']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Total Revenue", f"${rev:,.0f}", rev_g)
    with c2:
        render_kpi("Total Profit", f"${prof:,.0f}", prof_g)
    with c3:
        render_kpi("Total Orders", f"{orders:,}", ord_g)
    with c4:
        render_kpi("Items Sold", f"{sold:,}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Monthly Sales Trajectory ($)**")
        monthly = get_monthly_sales_trend(filtered_df)
        fig_trend = px.line(monthly, x='Month-Year', y='Sales', markers=True, color_discrete_sequence=[ACCENT])
        fig_trend.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Sales ($)", xaxis_title="")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        st.markdown("**Sales by Category**")
        cat_rev = get_category_revenue(filtered_df)
        fig_cat = px.pie(cat_rev, values='Sales', names='Category', hole=0.45, color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3])
        fig_cat.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("📄 View CEO Executive Summary Report", expanded=False):
        rep = get_ceo_report(filtered_df)
        st.text(rep)

    render_explainer(
        "Executive Dashboard Overview",
        "Provides business executives and store managers with a quick high-level summary of overall retail sales performance, net profitability, and operational metrics.",
        "Displays real-time KPI cards (Total Revenue, Total Profit, Total Orders, Items Sold), monthly sales trajectory trends over time, and revenue contribution percentages across Furniture, Office Supplies, and Technology."
    )


# =============================================================================
# 2. SALES ANALYTICS
# =============================================================================
elif selected_page == "📈 Sales Analytics":
    st.markdown('<div class="page-title">📈 Sales Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analyze sales patterns across time, product categories, and geographic regions.</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📅 Sales Trends", "🏷️ Categories & Sub-Categories", "🗺️ Regional Performance"])

    with t1:
        monthly = get_monthly_sales_trend(filtered_df)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly['Month-Year'], y=monthly['Sales'], name='Monthly Revenue ($)', marker_color=ACCENT))
        fig.add_trace(go.Scatter(x=monthly['Month-Year'], y=monthly['Profit'], name='Net Profit ($)', line=dict(color='#10b981', width=3)))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        sub_rev = get_sub_category_revenue(filtered_df)
        fig_sub = px.bar(sub_rev, x='Sales', y='Sub-Category', orientation='h', color='Category',
                         title="Revenue Generated by Each Sub-Category", color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3])
        fig_sub.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
        st.plotly_chart(fig_sub, use_container_width=True)

    with t3:
        reg_df = get_region_performance(filtered_df)
        c1, c2 = st.columns(2)
        with c1:
            fig_r = px.bar(reg_df, x='Region', y='Sales', color='Region', title="Total Sales by Region")
            fig_r.update_layout(height=340)
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            st.markdown("**Regional Profitability Table**")
            st.dataframe(reg_df[['Region', 'Sales', 'Profit', 'Quantity']], use_container_width=True)

    render_explainer(
        "Sales Analytics Overview",
        "Enables in-depth examination of revenue drivers, seasonal order patterns, and sales volume across product lines and geographic territories.",
        "Presents interactive monthly revenue vs. net profit graphs, horizontal revenue rankings for all 17 sub-categories (Phones, Chairs, Storage, etc.), and regional breakdowns comparing sales across West, East, Central, and South regions."
    )


# =============================================================================
# 3. ML SALES PREDICTION
# =============================================================================
elif selected_page == "🤖 ML Sales Prediction":
    st.markdown('<div class="page-title">🤖 Machine Learning Sales Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Predict the dollar sales of an order using Scikit-Learn regression models.</div>', unsafe_allow_html=True)

    sales_art = get_sales_models()
    lr_m = sales_art['lr_test_metrics']
    rf_m = sales_art['rf_test_metrics']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Random Forest R²", f"{rf_m['R2']:.3f}")
    with c2:
        render_kpi("Random Forest MAE", f"${rf_m['MAE']:.2f}")
    with c3:
        render_kpi("Linear Reg R²", f"{lr_m['R2']:.3f}")
    with c4:
        render_kpi("Linear Reg MAE", f"${lr_m['MAE']:.2f}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Actual Sales vs Predicted Sales (Test Data)**")
        comp = sales_art['comparison_df'].head(150)
        fig_c = px.scatter(comp, x='Actual', y='RF_Predicted', color='Category',
                           labels={'Actual': 'Actual Sales ($)', 'RF_Predicted': 'Predicted Sales ($)'},
                           title="Random Forest: Actual vs Predicted")
        fig_c.add_trace(go.Scatter(x=[0, 2000], y=[0, 2000], mode='lines', name='Ideal Line (y=x)', line=dict(dash='dash', color='#10b981')))
        fig_c.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

    with col2:
        st.markdown("**Top Feature Drivers (Random Forest)**")
        f_df = sales_art['feature_importance_df'].head(6)
        fig_f = px.bar(f_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=[ACCENT])
        fig_f.update_layout(yaxis={'categoryorder': 'total ascending'}, height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Try It Yourself: Interactive Order Price Predictor")
    st.caption("Change the order details below and click Predict to see the ML model forecast the sales value!")

    cat_opts = sales_art['cat_options']
    sub_map = sales_art['subcat_by_cat']

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        in_cat = st.selectbox("Category", cat_opts['Category'], key="in_cat")
        in_sub = st.selectbox("Sub-Category", sub_map[in_cat], key="in_sub")
    with i2:
        in_reg = st.selectbox("Region", cat_opts['Region'], key="in_reg")
        in_seg = st.selectbox("Segment", cat_opts['Segment'], key="in_seg")
    with i3:
        in_qty = st.number_input("Order Quantity", min_value=1, max_value=50, value=3, key="in_qty")
        in_disc = st.slider("Discount Applied", 0.0, 0.8, 0.1, 0.05, key="in_disc")
    with i4:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🚀 Predict Sales Value", type="primary", use_container_width=True)

    if predict_clicked:
        pred_res = predict_sales_single(sales_art, in_qty, in_disc, 11, in_cat, in_sub, in_reg, in_seg)
        rf_price = pred_res['rf_prediction']
        lr_price = pred_res['lr_prediction']
        
        p1, p2, p3 = st.columns(3)
        with p1:
            render_kpi("Random Forest Prediction", f"${rf_price:,.2f}")
        with p2:
            render_kpi("Linear Reg Prediction", f"${lr_price:,.2f}")
        with p3:
            est_prof = rf_price * 0.18
            render_kpi("Estimated Profit (~18%)", f"${est_prof:,.2f}")

    render_explainer(
        "ML Sales Prediction Overview",
        "Uses trained supervised machine learning algorithms to estimate the exact dollar revenue of retail orders before checkout.",
        "Compares Linear Regression against Random Forest Regressor using R² scores and Mean Absolute Error (MAE), shows actual vs. predicted test plots, feature importance rankings, and provides an interactive tool to predict order values based on quantity, discount, category, and region."
    )


# =============================================================================
# 4. DEMAND FORECASTING
# =============================================================================
elif selected_page == "🔮 Demand Forecasting":
    st.markdown('<div class="page-title">🔮 Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Predict future product demand to avoid stockouts and overstocking.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        horizon = st.selectbox("Forecast Ahead", ["Next 3 Months", "Next 6 Months", "Next 12 Months"])
        num_periods = 3 if "3" in horizon else (6 if "6" in horizon else 12)
    with c2:
        conf = st.selectbox("Confidence Level", ["95% Confidence (Standard)", "90% Confidence", "80% Confidence"])

    combined, future_df, recs = get_forecast(filtered_df, periods=num_periods)

    if not future_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            next_m = future_df.iloc[0]
            render_kpi("Next Month Demand", f"{next_m['Forecast']:.0f} units")
        with col2:
            tot_f = future_df['Forecast'].sum()
            render_kpi(f"{num_periods}-Month Cumulative Demand", f"{tot_f:.0f} units")
        with col3:
            render_kpi("Forecast Method", "Chronological ML Regressor")

        # Clean Line Chart
        st.markdown("**Historical Sales vs Future Demand Forecast**")
        fig_fc = go.Figure()
        actuals = combined[combined['Type'] == 'Actual']
        forecasts = combined[combined['Type'] == 'Forecast']

        fig_fc.add_trace(go.Scatter(x=actuals['Month-Year'], y=actuals['Value'], mode='lines+markers', name='Past Actual Sales',
                                    line=dict(color=ACCENT, width=2.5)))
        fig_fc.add_trace(go.Scatter(x=forecasts['Month-Year'], y=forecasts['Value'], mode='lines+markers', name='Predicted Future Demand',
                                    line=dict(color=ACCENT2, width=3, dash='dash'), marker=dict(size=8, symbol='diamond')))
        fig_fc.add_trace(go.Scatter(
            x=list(future_df['Month-Year']) + list(future_df['Month-Year'][::-1]),
            y=list(future_df['Upper CI']) + list(future_df['Lower CI'][::-1]),
            fill='toself', fillcolor='rgba(6, 182, 212, 0.15)', line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band'
        ))
        fig_fc.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Units Demanded", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("**Forecasted Values Table**")
        st.dataframe(future_df[['Month-Year', 'Forecast', 'Lower CI', 'Upper CI']].round(1), use_container_width=True)

    render_explainer(
        "Demand Forecasting Overview",
        "Forecasts future product demand volumes across upcoming months so store planners can order stock proactively and prevent stockouts.",
        "Plots historical monthly order quantities alongside future predictions for 3, 6, or 12 months, complete with upper and lower confidence intervals and a detailed forecasted quantities table."
    )


# =============================================================================
# 5. CUSTOMER SEGMENTATION
# =============================================================================
elif selected_page == "👥 Customer Segmentation":
    st.markdown('<div class="page-title">👥 Customer Segmentation (K-Means)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Group customers into actionable cohorts using RFM (Recency, Frequency, Monetary).</div>', unsafe_allow_html=True)

    km_art = get_kmeans_models()
    rfm = km_art['rfm']
    summary = km_art['segment_summary']

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("High-Value Customers", f"{summary.iloc[0]['Customer_Count']}", prefix="")
    with c2:
        render_kpi("Regular Customers", f"{summary.iloc[1]['Customer_Count']}", prefix="")
    with c3:
        render_kpi("Low-Engagement Customers", f"{summary.iloc[2]['Customer_Count']}", prefix="")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Customer Group Distribution**")
        fig_p = px.pie(summary, values='Customer_Count', names='Segment', hole=0.45,
                       color_discrete_sequence=['#10b981', ACCENT, '#ef4444'])
        fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_p, use_container_width=True)

    with col2:
        st.markdown("**Average Spend by Customer Segment ($)**")
        fig_b = px.bar(summary, x='Segment', y='Mean_Spend', color='Segment',
                       color_discrete_sequence=['#10b981', ACCENT, '#ef4444'])
        fig_b.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Average Spend ($)", xaxis_title="")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("**Customer Cohort Characteristics**")
    st.dataframe(summary[['Segment', 'Customer_Count', 'Customer_Share (%)', 'Mean_Spend', 'Total_Revenue', 'Mean_Recency_Days']], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Search Any Customer")
    c_names = sorted(rfm['Customer Name'].unique().tolist())
    picked_c = st.selectbox("Select Customer Name", c_names)
    c_info = rfm[rfm['Customer Name'] == picked_c].iloc[0]

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Group", c_info['Segment'])
    with q2:
        st.metric("Total Spent", f"${c_info['Monetary']:,.2f}")
    with q3:
        st.metric("Orders Placed", f"{int(c_info['Frequency'])} orders")
    with q4:
        st.metric("Last Order", f"{int(c_info['Recency'])} days ago")

    render_explainer(
        "Customer Segmentation Overview",
        "Groups shoppers into distinct customer segments using the K-Means Clustering algorithm based on RFM (Recency, Frequency, and Monetary value) analysis.",
        "Displays customer counts and proportions across High-Value, Regular, and Low-Engagement cohorts, average spending comparisons, detailed segment statistics, and an individual customer lookup tool to inspect any customer's buying habits."
    )


# =============================================================================
# 6. INVENTORY OPTIMIZATION
# =============================================================================
elif selected_page == "📦 Inventory Optimization":
    st.markdown('<div class="page-title">📦 Inventory Optimization & Stock Policies</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Data-driven Safety Stock and Reorder Points (ROP) to prevent stockouts.</div>', unsafe_allow_html=True)

    # Simplified policy slider
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        lead_time = st.slider("Supplier Delivery Lead Time (Days)", 3, 21, 7)
    with c_p2:
        serv_lvl = st.selectbox("Target Customer Service Level", [0.90, 0.95, 0.99], index=1,
                                format_func=lambda x: f"{int(x*100)}% (Standard Retail Protection)")

    inv_opt = calculate_inventory_optimization(filtered_df, lead_time_days=lead_time, service_level=serv_lvl)

    crit = len(inv_opt[inv_opt['Stock_Status'] == 'Critical (Stockout Risk)'])
    reord = len(inv_opt[inv_opt['Stock_Status'] == 'Reorder Required'])
    hlth = len(inv_opt[inv_opt['Stock_Status'] == 'Healthy Stock'])
    over = len(inv_opt[inv_opt['Stock_Status'] == 'Overstocked'])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Healthy Buffer", f"{hlth} items")
    with k2:
        render_kpi("Reorder Needed", f"{reord} items")
    with k3:
        render_kpi("Critical Stockout Risk", f"{crit} items")
    with k4:
        render_kpi("Overstocked Items", f"{over} items")

    st.markdown("**Stock Reorder Recommendations Table**")
    reorder_table = inv_opt[inv_opt['Stock_Status'].isin(['Critical (Stockout Risk)', 'Reorder Required'])].head(15)
    if not reorder_table.empty:
        st.dataframe(reorder_table[['Product Name', 'Category', 'Daily_Demand', 'Safety_Stock', 'Reorder_Point', 'Simulated_Current_Stock', 'Recommended_Reorder_Qty', 'Stock_Status']], use_container_width=True)
    else:
        st.success("✅ All product inventory levels are currently in healthy shape!")

    render_explainer(
        "Inventory Optimization Overview",
        "Generates mathematical inventory replenishment rules and Safety Stock targets to prevent stockouts while avoiding excessive holding costs.",
        "Calculates daily demand rates, Safety Stock buffers, Reorder Points (ROP), and automated restocking quantities for every product catalog item based on supplier lead times and target service levels."
    )


# =============================================================================
# 7. PRODUCT PERFORMANCE
# =============================================================================
elif selected_page == "🏆 Product Performance":
    st.markdown('<div class="page-title">🏆 Product Performance & Portfolio Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Identify top moneymakers, volume drivers, and loss-making underperformers.</div>', unsafe_allow_html=True)

    all_p = get_product_revenue(filtered_df)
    top_p = all_p.head(8)
    bot_p = all_p.tail(8).iloc[::-1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Revenue Products**")
        fig_t = px.bar(top_p, x='Sales', y='Product Name', orientation='h', color_discrete_sequence=['#10b981'])
        fig_t.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_t, use_container_width=True)
    with c2:
        st.markdown("**Low Performing / Loss Products**")
        fig_b = px.bar(bot_p, x='Sales', y='Product Name', orientation='h', color_discrete_sequence=['#ef4444'])
        fig_b.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    render_explainer(
        "Product Performance Overview",
        "Assesses product-level profitability and sales volume to identify high-performing bestsellers and underperforming products.",
        "Displays the top revenue-generating items, highlights low-performing or negative-margin products, and provides clear visibility into which catalog items contribute most to business revenue."
    )


# =============================================================================
# 8. ML MODEL EVALUATION
# =============================================================================
elif selected_page == "🔬 ML Model Evaluation":
    st.markdown('<div class="page-title">🔬 Machine Learning Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Academic benchmarks, error metrics, and validation test results.</div>', unsafe_allow_html=True)

    sales_art = get_sales_models()
    lr_t = sales_art['lr_test_metrics']
    rf_t = sales_art['rf_test_metrics']

    st.markdown("### 1. Sales Prediction Model Comparison")
    comp_df = pd.DataFrame([
        {'Model Name': 'Linear Regression (Baseline)', 'R² Score (Higher is better)': f"{lr_t['R2']:.4f}", 'MAE (Average Error)': f"${lr_t['MAE']:.2f}", 'RMSE': f"${lr_t['RMSE']:.2f}"},
        {'Model Name': 'Random Forest Regressor (Ensemble)', 'R² Score (Higher is better)': f"{rf_t['R2']:.4f}", 'MAE (Average Error)': f"${rf_t['MAE']:.2f}", 'RMSE': f"${rf_t['RMSE']:.2f}"}
    ])
    st.dataframe(comp_df, use_container_width=True)

    # Simple bar comparison
    err_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'Average Prediction Error ($ MAE)': [lr_t['MAE'], rf_t['MAE']]
    })
    fig_err = px.bar(err_df, x='Model', y='Average Prediction Error ($ MAE)', color='Model',
                     color_discrete_sequence=[ACCENT, ACCENT2], title="Average Error in Dollars (Lower is Better)")
    fig_err.update_layout(height=280)
    st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("### 2. Customer Clustering Evaluation")
    km_art = get_kmeans_models()
    e1, e2 = st.columns(2)
    with e1:
        st.metric("Silhouette Score (Clustering Quality)", f"{km_art['silhouette_score']:.3f}", help="Score from -1 to +1 measuring how distinct the customer clusters are.")
    with e2:
        st.metric("Optimal Clusters Chosen (k)", "3 Clusters (High-Value, Regular, At-Risk)")

    render_explainer(
        "ML Model Evaluation Overview",
        "Presents formal evaluation metrics and comparative benchmarks to measure the predictive quality of all trained machine learning models on unseen test data.",
        "Displays side-by-side model comparison tables showing R² score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), an error comparison chart, and the K-Means clustering Silhouette Score."
    )


# =============================================================================
# 9. AI RETAIL ASSISTANT
# =============================================================================
elif selected_page == "💬 AI Retail Assistant":
    st.markdown('<div class="page-title">💬 AI Retail Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ask questions about store sales, inventory, and predictions in natural language.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("**Quick Example Questions:**")
        prompts = [
            "Which products need restocking?",
            "What are the top selling products?",
            "Show customer segments",
            "What is the revenue summary?"
        ]
        for p in prompts:
            if st.button(p, key=f"btn_{p[:12]}", use_container_width=True):
                st.session_state.chat_input = p

    with c1:
        user_q = st.text_input("Type your question:", value=st.session_state.chat_input, placeholder="e.g. Which products need restocking?")
        if user_q and (not st.session_state.chat_history or st.session_state.chat_history[-1].get('user') != user_q):
            st.session_state.chat_history.append({'user': user_q})
            ans = get_ai_response(filtered_df, user_q)
            st.session_state.chat_history.append({'bot': ans})
            st.session_state.chat_input = ""
            st.rerun()

        for msg in st.session_state.chat_history:
            if 'user' in msg:
                st.markdown(f"**👤 You:** {msg['user']}")
            if 'bot' in msg:
                st.markdown(f"**🤖 Assistant:**\n{msg['bot']}")
            st.markdown("---")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    render_explainer(
        "AI Retail Assistant Overview",
        "Provides an interactive conversational assistant that allows store associates and managers to query retail metrics and stock alerts using natural language.",
        "A question-and-answer interface with quick-action prompts that dynamically queries real store sales, inventory alerts, top products, and customer cohorts without requiring manual SQL or database lookups."
    )
