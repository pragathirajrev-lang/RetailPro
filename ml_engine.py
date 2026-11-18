"""
ml_engine.py - Core Machine Learning Engine for Retail & Inventory Analytics
=============================================================================
Implements real, scikit-learn based machine learning workflows:
1. Sales Prediction (Linear Regression & Random Forest Regressor)
2. Demand Forecasting (Chronological Time-Series Feature Engineering & Regression)
3. Customer Segmentation (RFM Analysis & K-Means Clustering with Silhouette Score)
4. Inventory Optimization (Safety Stock, Reorder Point, EOQ, Simulated Stock Simulation)
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SALES_MODEL_PATH = os.path.join(MODELS_DIR, "sales_models.joblib")
FORECAST_MODEL_PATH = os.path.join(MODELS_DIR, "forecast_model.joblib")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")


# ============================================================================
# 1. SALES PREDICTION ENGINE
# ============================================================================

def get_sales_features_df(df):
    """
    Extracts features and target for sales prediction.
    Features: Quantity, Discount, Month, Category, Sub-Category, Region, Segment.
    Target: Sales
    """
    data = df.dropna(subset=['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Region', 'Segment']).copy()
    data['Month'] = pd.to_datetime(data['Order Date']).dt.month
    
    feature_cols = ['Quantity', 'Discount', 'Month', 'Category', 'Sub-Category', 'Region', 'Segment']
    X = data[feature_cols]
    y = data['Sales'].astype(float)
    return X, y, data


def train_sales_models(df, force_retrain=False):
    """
    Trains Linear Regression and Random Forest Regressor to predict Sales.
    Evaluates both on Train and Test sets (R2, MAE, RMSE).
    Saves trained pipelines and evaluation metrics to joblib.
    """
    if not force_retrain and os.path.exists(SALES_MODEL_PATH):
        try:
            return joblib.load(SALES_MODEL_PATH)
        except Exception:
            pass

    X, y, full_data = get_sales_features_df(df)
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = ['Quantity', 'Discount', 'Month']
    cat_cols = ['Category', 'Sub-Category', 'Region', 'Segment']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # 1. Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)

    # Helper function for metrics
    def calc_metrics(y_true, y_pred):
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {'R2': r2, 'MAE': mae, 'RMSE': rmse}

    # Evaluate Linear Regression
    lr_train_pred = lr_pipeline.predict(X_train)
    lr_test_pred = lr_pipeline.predict(X_test)
    lr_train_metrics = calc_metrics(y_train, lr_train_pred)
    lr_test_metrics = calc_metrics(y_test, lr_test_pred)

    # Evaluate Random Forest
    rf_train_pred = rf_pipeline.predict(X_train)
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_train_metrics = calc_metrics(y_train, rf_train_pred)
    rf_test_metrics = calc_metrics(y_test, rf_test_pred)

    # Feature Importance extraction from Random Forest
    rf_model = rf_pipeline.named_steps['regressor']
    ohe_step = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe_step.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names
    importances = rf_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Sample comparison dataframe for visual inspection
    comparison_df = pd.DataFrame({
        'Actual': y_test.iloc[:500].values,
        'LR_Predicted': lr_test_pred[:500],
        'RF_Predicted': rf_test_pred[:500],
        'Quantity': X_test['Quantity'].iloc[:500].values,
        'Category': X_test['Category'].iloc[:500].values,
        'Sub-Category': X_test['Sub-Category'].iloc[:500].values
    })

    feature_cols = X.columns.tolist()
    model_artifacts = {
        'lr_pipeline': lr_pipeline,
        'rf_pipeline': rf_pipeline,
        'lr_train_metrics': lr_train_metrics,
        'lr_test_metrics': lr_test_metrics,
        'rf_train_metrics': rf_train_metrics,
        'rf_test_metrics': rf_test_metrics,
        'feature_importance_df': feature_importance_df,
        'comparison_df': comparison_df,
        'y_test_sample': y_test.values[:300],
        'rf_test_pred_sample': rf_test_pred[:300],
        'lr_test_pred_sample': lr_test_pred[:300],
        'feature_cols': feature_cols,
        'cat_options': {col: sorted(full_data[col].unique().tolist()) for col in cat_cols},
        'subcat_by_cat': {cat: sorted(full_data[full_data['Category'] == cat]['Sub-Category'].unique().tolist()) for cat in full_data['Category'].unique()},
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    joblib.dump(model_artifacts, SALES_MODEL_PATH)
    return model_artifacts


def predict_sales_single(model_artifacts, quantity, discount, month, category, sub_category, region, segment):
    """
    Generates real-time sales predictions for custom user inputs using both LR and RF.
    """
    input_df = pd.DataFrame([{
        'Quantity': float(quantity),
        'Discount': float(discount),
        'Month': int(month),
        'Category': str(category),
        'Sub-Category': str(sub_category),
        'Region': str(region),
        'Segment': str(segment)
    }])

    lr_pred = float(model_artifacts['lr_pipeline'].predict(input_df)[0])
    rf_pred = float(model_artifacts['rf_pipeline'].predict(input_df)[0])
    
    # Negative predictions are physically impossible for retail sales
    lr_pred = max(0.0, lr_pred)
    rf_pred = max(0.0, rf_pred)

    return {
        'lr_prediction': lr_pred,
        'rf_prediction': rf_pred,
        'recommended_prediction': rf_pred
    }


# ============================================================================
# 2. DEMAND FORECASTING ENGINE (TIME-SERIES ML WITH CHRONOLOGICAL SPLIT)
# ============================================================================

def build_time_series_features(df, freq='M'):
    """
    Aggregates data by Month ('M') or Week ('W') and builds chronological lag & seasonal features.
    """
    data = df.copy()
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data = data.sort_values('Order Date')

    if freq == 'M':
        ts = data.groupby(pd.Grouper(key='Order Date', freq='ME')).agg({
            'Quantity': 'sum',
            'Sales': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        ts['Period'] = ts['Order Date'].dt.to_period('M').astype(str)
        ts['Seasonal_Month'] = ts['Order Date'].dt.month
        ts['Seasonal_Quarter'] = ts['Order Date'].dt.quarter
    else:
        ts = data.groupby(pd.Grouper(key='Order Date', freq='W-MON')).agg({
            'Quantity': 'sum',
            'Sales': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        ts['Period'] = ts['Order Date'].dt.strftime('%Y-W%U')
        ts['Seasonal_Month'] = ts['Order Date'].dt.month
        ts['Seasonal_Quarter'] = ts['Order Date'].dt.quarter

    ts['Time_Index'] = np.arange(len(ts))
    
    # Lag features
    ts['Lag_1'] = ts['Quantity'].shift(1)
    ts['Lag_2'] = ts['Quantity'].shift(2)
    ts['Lag_3'] = ts['Quantity'].shift(3)
    
    # Rolling averages
    ts['Rolling_Mean_3'] = ts['Quantity'].shift(1).rolling(window=3).mean()
    ts['Rolling_Std_3'] = ts['Quantity'].shift(1).rolling(window=3).std().fillna(0)

    # Drop initial NaN periods created by lag creation
    ts_clean = ts.dropna().reset_index(drop=True)
    return ts_clean


def train_demand_forecast_model(df, freq='M', force_retrain=False):
    """
    Trains a time-series regression model using CHRONOLOGICAL train-test split.
    Prevents data leakage by strictly training on past periods and testing on subsequent periods.
    """
    ts_data = build_time_series_features(df, freq=freq)
    
    if len(ts_data) < 6:
        return None

    feature_cols = ['Time_Index', 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Seasonal_Month', 'Seasonal_Quarter']
    X = ts_data[feature_cols]
    y = ts_data['Quantity'].values

    # CHRONOLOGICAL SPLIT: strictly first 80% as train, last 20% as out-of-time test
    split_idx = int(len(ts_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    periods_train = ts_data['Period'].iloc[:split_idx].values
    periods_test = ts_data['Period'].iloc[split_idx:].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Regressor for non-linear trend & seasonality
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_r2 = float(r2_score(y_train, train_pred))
    train_mae = float(mean_absolute_error(y_train, train_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))

    test_r2 = float(r2_score(y_test, test_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    # Compute residual standard error for confidence intervals
    residuals = y_train - train_pred
    residual_std = float(np.std(residuals))

    # Multi-step future recursive forecasting for next 6 periods
    last_row = ts_data.iloc[-1].copy()
    future_forecasts = []
    
    curr_lags = [ts_data['Quantity'].iloc[-1], ts_data['Quantity'].iloc[-2], ts_data['Quantity'].iloc[-3]]
    last_date = pd.to_datetime(ts_data['Order Date'].iloc[-1])

    for step in range(1, 13):
        if freq == 'M':
            next_date = last_date + pd.DateOffset(months=step)
            next_period = next_date.strftime('%Y-%m')
        else:
            next_date = last_date + pd.DateOffset(weeks=step)
            next_period = next_date.strftime('%Y-W%U')

        next_time_index = len(ts_data) + step - 1
        lag1, lag2, lag3 = curr_lags[0], curr_lags[1], curr_lags[2]
        roll_mean = np.mean([lag1, lag2, lag3])
        s_month = next_date.month
        s_quarter = next_date.quarter

        feat_vector = pd.DataFrame([{
            'Time_Index': next_time_index,
            'Lag_1': lag1,
            'Lag_2': lag2,
            'Lag_3': lag3,
            'Rolling_Mean_3': roll_mean,
            'Seasonal_Month': s_month,
            'Seasonal_Quarter': s_quarter
        }])[feature_cols]

        feat_scaled = scaler.transform(feat_vector)
        pred_qty = float(model.predict(feat_scaled)[0])
        pred_qty = max(0.0, pred_qty)

        # Confidence interval expands slightly with step horizon
        ci_margin = 1.96 * residual_std * (1.0 + 0.05 * step)

        future_forecasts.append({
            'Period': next_period,
            'Order Date': next_date,
            'Predicted_Demand': pred_qty,
            'Lower_CI': max(0.0, pred_qty - ci_margin),
            'Upper_CI': pred_qty + ci_margin
        })

        # Update lags recursively
        curr_lags = [pred_qty, curr_lags[0], curr_lags[1]]

    future_df = pd.DataFrame(future_forecasts)

    # Category and Product Demand Trends
    cat_trends = analyze_demand_trends(df, level='Category')
    prod_trends = analyze_demand_trends(df, level='Product Name', top_n=15)

    artifacts = {
        'model': model,
        'scaler': scaler,
        'ts_data': ts_data,
        'train_metrics': {'R2': train_r2, 'MAE': train_mae, 'RMSE': train_rmse},
        'test_metrics': {'R2': test_r2, 'MAE': test_mae, 'RMSE': test_rmse},
        'periods_train': periods_train,
        'periods_test': periods_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'future_df': future_df,
        'residual_std': residual_std,
        'cat_trends': cat_trends,
        'prod_trends': prod_trends,
        'freq': freq
    }
    return artifacts


def analyze_demand_trends(df, level='Category', top_n=None):
    """
    Computes demand velocity and trend direction (increasing vs decreasing) for categories or products.
    """
    data = df.copy()
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    max_date = data['Order Date'].max()
    mid_date = max_date - pd.DateOffset(months=6)

    recent_df = data[data['Order Date'] >= mid_date]
    prior_df = data[data['Order Date'] < mid_date]

    recent_qty = recent_df.groupby(level)['Quantity'].sum()
    prior_qty = prior_df.groupby(level)['Quantity'].sum()

    trends = pd.DataFrame({'Recent_6M_Demand': recent_qty, 'Prior_Demand': prior_qty}).fillna(0)
    
    # Normalize prior demand by duration ratio if available
    recent_months = max(1, (max_date.year - mid_date.year)*12 + max_date.month - mid_date.month)
    min_date = data['Order Date'].min()
    prior_months = max(1, (mid_date.year - min_date.year)*12 + mid_date.month - min_date.month)
    
    trends['Prior_6M_Equiv'] = trends['Prior_Demand'] * (recent_months / prior_months)
    trends['Growth_Rate (%)'] = np.where(
        trends['Prior_6M_Equiv'] > 0,
        ((trends['Recent_6M_Demand'] - trends['Prior_6M_Equiv']) / trends['Prior_6M_Equiv']) * 100,
        0.0
    )

    trends['Trend_Status'] = trends['Growth_Rate (%)'].apply(
        lambda x: 'Surging Demand' if x >= 15 else ('Declining Demand' if x <= -15 else 'Stable Demand')
    )
    
    trends = trends.sort_values('Growth_Rate (%)', ascending=False).reset_index()
    if top_n:
        return trends.head(top_n)
    return trends


# ============================================================================
# 3. CUSTOMER SEGMENTATION ENGINE (K-MEANS CLUSTERING & RFM)
# ============================================================================

def compute_rfm(df):
    """
    Computes Recency, Frequency, and Monetary (RFM) values for each unique customer.
    """
    data = df.copy()
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    reference_date = data['Order Date'].max() + pd.Timedelta(days=1)

    rfm = data.groupby('Customer ID').agg({
        'Order Date': lambda x: (reference_date - x.max()).days,
        'Order ID': 'nunique',
        'Sales': 'sum',
        'Profit': 'sum',
        'Customer Name': 'first'
    }).reset_index()

    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Total_Profit', 'Customer Name']
    rfm['Avg_Order_Value'] = (rfm['Monetary'] / rfm['Frequency']).round(2)
    return rfm


def train_customer_segmentation(df, n_clusters=3, force_retrain=False):
    """
    Trains K-Means clustering on RFM features with standard scaling.
    Computes Silhouette Score and sorts cluster labels logically:
    - High-Value Customers (Highest Monetary & Frequency)
    - Regular Customers (Moderate Spend & Regular Orders)
    - Low-Engagement Customers (Low Frequency / High Recency)
    """
    if not force_retrain and os.path.exists(KMEANS_MODEL_PATH):
        try:
            return joblib.load(KMEANS_MODEL_PATH)
        except Exception:
            pass

    rfm = compute_rfm(df)

    # Log transformation for right-skewed RFM features
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Calculate metrics
    inertia = float(kmeans.inertia_)
    sil_score = float(silhouette_score(scaled_features, cluster_labels))

    # Calculate Elbow Curve data (k=2 to k=6) for visual academic demonstration
    elbow_data = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        sc = float(silhouette_score(scaled_features, km.labels_))
        elbow_data.append({'k': k, 'Inertia': float(km.inertia_), 'Silhouette_Score': sc})

    elbow_df = pd.DataFrame(elbow_data)

    rfm['Cluster_Raw'] = cluster_labels

    # Sort clusters by average monetary value to ensure consistent interpretable segment naming
    cluster_ranks = rfm.groupby('Cluster_Raw')['Monetary'].mean().sort_values(ascending=False).index.tolist()
    
    if n_clusters == 3:
        mapping = {
            cluster_ranks[0]: 'High-Value Customers',
            cluster_ranks[1]: 'Regular Customers',
            cluster_ranks[2]: 'Low-Engagement Customers'
        }
    else:
        mapping = {
            cluster_ranks[0]: 'High-Value Customers',
            cluster_ranks[1]: 'Regular Customers',
            cluster_ranks[-1]: 'Low-Engagement Customers'
        }
        for i, c in enumerate(cluster_ranks[2:-1], start=3):
            mapping[c] = f'Segment {i}'

    rfm['Segment'] = rfm['Cluster_Raw'].map(mapping)

    # Segment Profile Summary
    segment_summary = rfm.groupby('Segment').agg({
        'Customer ID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'Total_Profit': 'mean',
        'Avg_Order_Value': 'mean'
    }).round(2)

    segment_summary.columns = [
        'Customer_Count', 'Mean_Recency_Days', 'Mean_Order_Count',
        'Mean_Spend', 'Total_Revenue', 'Mean_Profit', 'Mean_AOV'
    ]
    segment_summary = segment_summary.reset_index()
    total_rev = rfm['Monetary'].sum()
    segment_summary['Revenue_Share (%)'] = ((segment_summary['Total_Revenue'] / total_rev) * 100).round(1)
    segment_summary['Customer_Share (%)'] = ((segment_summary['Customer_Count'] / len(rfm)) * 100).round(1)

    artifacts = {
        'kmeans': kmeans,
        'scaler': scaler,
        'rfm': rfm,
        'inertia': inertia,
        'silhouette_score': sil_score,
        'elbow_df': elbow_df,
        'segment_summary': segment_summary,
        'mapping': mapping,
        'n_clusters': n_clusters,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    joblib.dump(artifacts, KMEANS_MODEL_PATH)
    return artifacts


# ============================================================================
# 4. INVENTORY OPTIMIZATION ENGINE (SAFETY STOCK, ROP & REORDER ALERTS)
# ============================================================================

def calculate_inventory_optimization(df, lead_time_days=7, service_level=0.95, holding_cost_pct=0.20, stock_multiplier=1.0):
    """
    Computes mathematical inventory recommendations based on historical demand:
    1. Average Daily Demand (d)
    2. Demand Standard Deviation (sigma_d)
    3. Safety Stock: SS = Z * sigma_d * sqrt(L)
    4. Reorder Point: ROP = (d * L) + SS
    5. Economic Order Quantity: EOQ = sqrt((2 * D * S) / H)
    6. Simulated Current Stock (clearly documented transparent simulation based on sales velocity)
    """
    # Z factor for service level
    z_table = {0.80: 1.28, 0.85: 1.44, 0.90: 1.645, 0.95: 1.96, 0.98: 2.05, 0.99: 2.33}
    z_val = z_table.get(service_level, 1.96)

    data = df.copy()
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    date_range_days = max(1, (data['Order Date'].max() - data['Order Date'].min()).days)
    latest_order_date = data['Order Date'].max()

    # Product-level daily demand aggregations
    prod_daily = data.groupby(['Product Name', 'Order Date'])['Quantity'].sum().reset_index()
    prod_stats = prod_daily.groupby('Product Name').agg(
        Total_Qty=('Quantity', 'sum'),
        Active_Days=('Order Date', 'nunique'),
        Std_Daily_Qty=('Quantity', 'std'),
        Mean_Daily_Qty=('Quantity', 'mean'),
        Last_Sale_Date=('Order Date', 'max')
    ).reset_index()

    prod_stats['Std_Daily_Qty'] = prod_stats['Std_Daily_Qty'].fillna(0.5)

    # Merge product price, category, subcategory
    prod_meta = data.groupby('Product Name').agg({
        'Category': 'first',
        'Sub-Category': 'first',
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()

    inv_df = prod_stats.merge(prod_meta, on='Product Name', how='left')
    inv_df['Unit_Price'] = (inv_df['Sales'] / inv_df['Total_Qty'].replace(0, 1)).round(2)
    inv_df['Daily_Demand'] = inv_df['Total_Qty'] / date_range_days
    
    # 1. Safety Stock
    inv_df['Safety_Stock'] = np.ceil(z_val * inv_df['Std_Daily_Qty'] * np.sqrt(lead_time_days)).astype(int)
    inv_df['Safety_Stock'] = inv_df['Safety_Stock'].clip(lower=2)

    # 2. Reorder Point (ROP)
    inv_df['Lead_Time_Demand'] = inv_df['Daily_Demand'] * lead_time_days
    inv_df['Reorder_Point'] = np.ceil(inv_df['Lead_Time_Demand'] + inv_df['Safety_Stock']).astype(int)

    # 3. EOQ (Annual Demand D, Ordering Cost S assumed $50, Unit Holding Cost H)
    annual_demand = inv_df['Daily_Demand'] * 365
    ordering_cost = 50.0  # standard fixed order setup cost assumption
    holding_cost_per_unit = (inv_df['Unit_Price'] * holding_cost_pct).clip(lower=1.0)
    inv_df['EOQ'] = np.ceil(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)).astype(int)
    inv_df['EOQ'] = inv_df['EOQ'].clip(lower=5)

    # 4. Transparent Simulated Current Stock
    # Based on actual recent 30-day velocity scaled by user-adjustable stock_multiplier
    recent_30d = data[data['Order Date'] >= (latest_order_date - pd.Timedelta(days=30))]
    recent_qty = recent_30d.groupby('Product Name')['Quantity'].sum().to_dict()
    
    np.random.seed(42)
    simulated_stock = []
    for _, row in inv_df.iterrows():
        pname = row['Product Name']
        base_30d = recent_qty.get(pname, row['Daily_Demand'] * 30)
        # Add realistic inventory variance around reorder point & safety stock
        sim_val = max(1, int((base_30d * 0.8 + row['Safety_Stock'] * 1.2) * stock_multiplier))
        simulated_stock.append(sim_val)

    inv_df['Simulated_Current_Stock'] = simulated_stock
    inv_df['Days_Since_Last_Sale'] = (latest_order_date - inv_df['Last_Sale_Date']).dt.days

    # 5. Inventory Status Classification
    def classify_stock(row):
        curr = row['Simulated_Current_Stock']
        rop = row['Reorder_Point']
        ss = row['Safety_Stock']
        days_idle = row['Days_Since_Last_Sale']

        if days_idle > 180 and curr > 5:
            return 'Dead / Excess Stock'
        elif curr < ss:
            return 'Critical (Stockout Risk)'
        elif curr <= rop:
            return 'Reorder Required'
        elif curr > (rop * 2.5):
            return 'Overstocked'
        else:
            return 'Healthy Stock'

    inv_df['Stock_Status'] = inv_df.apply(classify_stock, axis=1)

    # 6. Restock Quantity Recommendation
    inv_df['Recommended_Reorder_Qty'] = np.where(
        inv_df['Simulated_Current_Stock'] <= inv_df['Reorder_Point'],
        np.maximum(inv_df['EOQ'], (inv_df['Reorder_Point'] * 2) - inv_df['Simulated_Current_Stock']),
        0
    ).astype(int)

    inv_df['Days_Of_Supply'] = np.where(
        inv_df['Daily_Demand'] > 0,
        (inv_df['Simulated_Current_Stock'] / inv_df['Daily_Demand']).round(1),
        999
    )

    inv_df['Inventory_Value'] = (inv_df['Simulated_Current_Stock'] * inv_df['Unit_Price']).round(2)

    return inv_df.sort_values('Simulated_Current_Stock')
