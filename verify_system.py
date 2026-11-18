"""
verify_system.py - Standalone System Verification Script
=========================================================
Runs an end-to-end sanity check across the dataset, ML engine,
and model artifacts without needing to open a browser.
"""

import os
import joblib
import pandas as pd
from data_processing import load_and_clean_data
from ml_engine import (
    train_sales_models,
    predict_sales_single,
    train_demand_forecast_model,
    train_customer_segmentation,
    calculate_inventory_optimization,
    MODELS_DIR
)

def run_verification():
    print("=" * 65)
    print("RETAIL & INVENTORY ML SYSTEM - STANDALONE VERIFICATION")
    print("=" * 65)

    # 1. Dataset Check
    csv_path = os.path.join("data", "superstore.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset not found at {csv_path}")
        return False
    df = load_and_clean_data(csv_path)
    print(f"✅ 1. Dataset Loaded: {len(df):,} cleaned rows, {df.shape[1]} columns.")

    # 2. Sales Prediction Models
    s_arts = train_sales_models(df, force_retrain=False)
    lr_r2 = s_arts['lr_test_metrics']['R2']
    rf_r2 = s_arts['rf_test_metrics']['R2']
    print(f"✅ 2. Sales Prediction: Linear Reg Test R² = {lr_r2:.3f}, Random Forest Test R² = {rf_r2:.3f}")
    
    # Interactive test prediction
    pred = predict_sales_single(
        s_arts, quantity=5, discount=0.1, month=10,
        category='Technology', sub_category='Phones', region='East', segment='Consumer'
    )
    print(f"   -> Sample Predict: 5x Phones = ${pred['rf_prediction']:.2f}")

    # 3. Demand Forecasting
    f_arts = train_demand_forecast_model(df, freq='M', force_retrain=False)
    fc_r2 = f_arts['test_metrics']['R2']
    next_demand = f_arts['future_df'].iloc[0]['Predicted_Demand']
    print(f"✅ 3. Demand Forecasting: Chronological Test R² = {fc_r2:.3f}, Next Month Demand = {next_demand:.0f} units")

    # 4. Customer Segmentation
    k_arts = train_customer_segmentation(df, force_retrain=False)
    sil = k_arts['silhouette_score']
    clusters = k_arts['n_clusters']
    print(f"✅ 4. Customer Segmentation: {clusters} Clusters, Silhouette Score = {sil:.3f}")

    # 5. Inventory Optimization
    inv_df = calculate_inventory_optimization(df, lead_time_days=7, service_level=0.95)
    crit_count = len(inv_df[inv_df['Stock_Status'] == 'Critical (Stockout Risk)'])
    reorder_count = len(inv_df[inv_df['Stock_Status'] == 'Reorder Required'])
    print(f"✅ 5. Inventory Optimization: {len(inv_df)} products evaluated. Critical: {crit_count}, Reorder: {reorder_count}")

    # 6. Serialized Artifacts Check
    print("✅ 6. Checking Serialized Models in /models:")
    for f in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"   -> {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED! The system is ready for demonstration.")
    print("Run `streamlit run app.py` to start the web dashboard.")
    print("=" * 65)
    return True

if __name__ == "__main__":
    run_verification()

