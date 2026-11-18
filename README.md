# RetailPro AI: Machine Learning Retail & Inventory Management System 🛒📊

> **A Complete, End-to-End Enterprise Retail Analytics, Demand Forecasting, Customer Segmentation, and Inventory Optimization Platform Powered by Machine Learning.**  
> *Developed for Academic & B.Tech Computer Science and Engineering Project Demonstrations.*

---

## 📌 1. Project Overview

**RetailPro AI** is an intelligent, data-driven retail management system built with Python, Scikit-learn, and Streamlit. The platform moves beyond static reporting by integrating genuine Machine Learning algorithms to:
1. **Predict future sales** based on historical pricing, discounts, and order features.
2. **Forecast product demand** using chronological time-series machine learning models without look-ahead data leakage.
3. **Segment customers** using RFM (Recency, Frequency, Monetary) analytics and unsupervised **K-Means Clustering**.
4. **Optimize inventory stock levels** by calculating mathematical **Safety Stock**, **Reorder Points (ROP)**, and **Economic Order Quantity (EOQ)**.
5. **Classify product performance** via a 4-quadrant volume-vs-profitability matrix.
6. **Evaluate and benchmark models** rigorously using $R^2$, MAE, RMSE, Silhouette Scores, and error residual analysis.

---

## 🛠️ 2. Technology Stack

- **Core Language**: Python 3.10+
- **Frontend Dashboard**: Streamlit (Modern responsive UI with Dark / Light theme support)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (`LinearRegression`, `RandomForestRegressor`, `KMeans`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer`)
- **Data Visualization**: Plotly Express & Plotly Graph Objects (Interactive 2D & 3D charts)
- **Model Serialization**: Joblib (`.joblib` cached pipeline artifacts)
- **Report Generation**: FPDF, XlsxWriter, OpenPyXL

---

## 📂 3. Project Directory Architecture

```
Retail-and-Inventory-Analysis-main/
│
├── app.py                      # Main Streamlit web application & routing
├── ml_engine.py                # Core ML engine (Training, Inference, Metrics, Pipelines)
├── data_processing.py          # Data ingestion, cleaning, KPI aggregations & business logic
├── requirements.txt            # Python dependencies with version specifications
├── README.md                   # Comprehensive project documentation & demo guide
│
├── data/
│   ├── superstore.csv          # Primary retail dataset (9,994 transactional records)
│   └── simple_retail_data.csv  # Supplementary retail dataset
│
└── models/                     # Serialized machine learning models (.joblib)
    ├── sales_models.joblib     # Trained Linear Regression & Random Forest pipelines
    ├── forecast_model.joblib   # Trained time-series demand forecasting model
    └── kmeans_model.joblib     # Trained K-Means customer segmentation artifacts
```

---

## 🧠 4. Machine Learning Algorithms & Implementation Details

### 1. Sales Prediction (Supervised Regression)
- **Algorithms**: **Linear Regression** vs. **Random Forest Regressor** (Ensemble of 100 Decision Trees).
- **Features Used**: `Quantity`, `Discount`, `Month`, `Category`, `Sub-Category`, `Region`, `Segment`.
- **Preprocessing Pipeline**:
  - Numerical attributes: `StandardScaler`
  - Categorical attributes: `OneHotEncoder(handle_unknown='ignore')`
  - Encapsulated into a Scikit-learn `Pipeline` to prevent train-test contamination.
- **Train/Test Split**: 80% Training, 20% Testing (`random_state=42`).
- **Evaluation Metrics**:
  - $R^2$ Score (Coefficient of Determination)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- **Interactive Predictor**: Real-time interactive inference tool allowing users to enter custom quantity, discount, category, and region to generate side-by-side predictions from both models.

### 2. Time-Series Demand Forecasting
- **Approach**: Chronological feature engineering + Non-linear Regression.
- **Features Created**:
  - Auto-regressive lag features: $t-1, t-2, t-3$
  - Rolling moving averages: 3-period rolling mean and standard deviation
  - Temporal trend index ($0, 1, 2, \dots$)
  - Calendar month & quarter seasonal indicators
- **Chronological Split**: Strictly partitions the timeline chronologically (first 80% historical periods for training, subsequent 20% unseen periods for test validation). Eliminates temporal look-ahead bias.
- **Recursive Multi-Step Projection**: Recursively projects future demand 3 to 12 periods ahead with expanding confidence interval bands ($\pm 1.96 \times \sigma_{\epsilon}$).
- **Demand Velocity Detector**: Classifies categories and products into **Surging Demand** ($\ge +15\%$), **Stable**, or **Declining Demand** ($\le -15\%$).

### 3. Customer Segmentation (Unsupervised Clustering)
- **Algorithm**: **K-Means Clustering** ($k=3$).
- **Features (RFM Analysis)**:
  - **Recency ($R$)**: Days since customer's most recent purchase ($D_{\max} - D_{\text{customer}}$).
  - **Frequency ($F$)**: Count of unique orders placed by the customer.
  - **Monetary ($M$)**: Total revenue spend across all customer transactions.
- **Normalization**: Log-transformation (`np.log1p`) followed by `StandardScaler` to remove skewness.
- **Validation**: Silhouette Score (`silhouette_score`) and Inertia with an Elbow curve ($k=2$ to $k=6$).
- **Segment Profiles**:
  - **High-Value Customers (Champions)**: Low recency, high frequency, high spend.
  - **Regular Customers (Loyal)**: Moderate recency and frequency, steady spend.
  - **Low-Engagement Customers (At-Risk)**: High recency (long inactivity), low frequency.

### 4. Mathematical Inventory Optimization & Replenishment
- **Formulas & Assumptions**:
  - **Daily Demand Rate ($\bar{d}$)**: $\frac{\text{Total Quantity Sold}}{\text{Active Days}}$
  - **Safety Stock ($\text{SS}$)**: $\text{SS} = Z \times \sigma_d \times \sqrt{L}$
    *(where $Z$ is Service Level factor, $\sigma_d$ is standard deviation of daily demand, and $L$ is lead time)*
  - **Reorder Point ($\text{ROP}$)**: $\text{ROP} = (\bar{d} \times L) + \text{SS}$
  - **Economic Order Quantity ($\text{EOQ}$)**: $\text{EOQ} = \sqrt{\frac{2 \times D \times S}{H}}$
- **Simulation Disclaimer**: The Superstore dataset tracks retail sales orders rather than live physical warehouse IoT sensors. The system transparently models stock levels from historical sales velocity and provides **interactive warehouse policy controls** (Lead Time, Service Level, Holding Cost %, Stock Multiplier).
- **Categorization**:
  - 🔴 **Critical (Stockout Risk)**: Stock < Safety Stock
  - 🟡 **Reorder Required**: Stock $\le$ Reorder Point
  - 🟢 **Healthy Stock**: ROP < Stock $\le 2 \times \text{ROP}$
  - 🔵 **Overstocked**: Stock $> 2.5 \times \text{ROP}$
  - ⚫ **Dead / Excess Stock**: Days since last order $> 180$ days

---

## 🖥️ 5. Streamlit Dashboard Sections

1. **Executive Dashboard**: Executive KPIs, revenue/profit trajectory, regional share, automated restock alerts, and downloadable CEO report (PDF & Text).
2. **Sales Analytics**: Deep-dive tabs for time trends, category performance, US state choropleth map, order value distributions, and discount margin erosion scatter.
3. **ML Sales Prediction**: Model comparison table, actual vs predicted scatter with $y=x$ ideal line, feature importance bar chart, and what-if prediction tool.
4. **Demand Forecasting**: Interactive timeline chart showing historical actuals, out-of-time test validation, and future forecast with confidence bands; demand momentum detector.
5. **Customer Segmentation**: Interactive 3D RFM scatter plot, segment summary tables, retention strategy playbook, and single customer lookup tool.
6. **Inventory Optimization**: Interactive simulation sliders (lead time, service level), reorder recommendation table, Excel export, and dead stock exposure analysis.
7. **Product Performance**: Best sellers vs bottom performers, 4-quadrant product portfolio matrix (Stars, Cash Cows, Niche, Underperformers).
8. **ML Model Evaluation**: Academic evaluation showing train vs test metrics, residual error distributions, chronological split proof, and a one-click "Retrain All Models" button.
9. **AI Retail Assistant**: Chatbot interface answering natural language questions using live dataset computations and ML predictions.

---

## ⚡ 6. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git (optional, for cloning)

### Step 1: Clone or Navigate to the Project Directory
```bash
cd Retail-and-Inventory-Analysis-main
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🎓 7. B.Tech College Project Demonstration Walkthrough

When demonstrating this project for academic evaluation or viva voce, follow this recommended sequence:

| Step | Section | Key Discussion Points |
|---|---|---|
| **1** | **Executive Dashboard** | Show real-time KPI computation from 9,994 records; explain growth rate calculations and automated alert indicators. |
| **2** | **ML Sales Prediction** | Compare Linear Regression vs. Random Forest. Point out why Random Forest captures non-linear pricing effects. Input values into the interactive predictor and show instantaneous inference. |
| **3** | **Demand Forecasting** | Explain why **chronological splitting** was used instead of random splitting to prevent data leakage in time series. Highlight the out-of-time test validation and future confidence intervals. |
| **4** | **Customer Segmentation** | Rotate the **3D RFM scatter plot**. Explain how Recency, Frequency, and Monetary features were log-scaled and normalized, and how $k=3$ was validated using Silhouette Scores. |
| **5** | **Inventory Optimization** | Demonstrate the mathematical Safety Stock ($SS$) and Reorder Point ($ROP$) formulas. Move the **Lead Time** and **Service Level** sliders to show dynamic recalculation of replenishment quantities. |
| **6** | **ML Model Evaluation** | Walk through the train vs test metrics table ($R^2$, MAE, RMSE) and explain the bias-variance tradeoff and residual error distributions. Click the **"Retrain All ML Models"** button to show real-time training and Joblib serialization. |

---

## 📄 8. License & Acknowledgments
- **Dataset**: Superstore Retail Transaction Dataset
- **Frameworks**: Streamlit, Scikit-learn, Plotly, Pandas
- *Designed and developed for academic excellence and retail analytics innovation.*
