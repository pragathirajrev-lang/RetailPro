"""
Generates the comprehensive Project Implementation Plan & Technical Report PDF.
Suitable for Academic Report Submission and PPT Presentation preparation.
"""

from fpdf import FPDF
import datetime

class ProjectReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # Skip header on cover page
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(100, 116, 139) # slate
        self.cell(0, 8, 'RetailPro AI - Complete Implementation Plan & Project Technical Report', border=0, align='L')
        self.cell(0, 8, 'B.Tech CSE Project', border=0, align='R')
        self.ln(9)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, num, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 41, 59) # dark navy
        self.cell(0, 9, f"{num}. {title}", border=0, align='L')
        self.ln(10)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(79, 70, 229) # indigo
        self.cell(0, 7, title, border=0, align='L')
        self.ln(8)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(51, 65, 85) # slate dark
        self.multi_cell(0, 5.2, text)
        self.ln(2.5)

    def bullet_point(self, bold_prefix, text):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(30, 41, 59)
        self.write(5.2, f"  * {bold_prefix}: ")
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(51, 65, 85)
        self.write(5.2, f"{text}\n")
        self.ln(1.5)

    def key_value_box(self, title, items):
        self.set_fill_color(248, 250, 252) # light gray
        self.set_draw_color(203, 213, 225)
        self.rect(10, self.get_y(), 190, 8 + len(items) * 6, style='DF')
        curr_y = self.get_y() + 3
        self.set_xy(14, curr_y)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, title)
        self.ln(6)
        for k, v in items:
            self.set_x(14)
            self.set_font('Helvetica', 'B', 8.5)
            self.set_text_color(30, 41, 59)
            self.cell(45, 5, f"{k}:")
            self.set_font('Helvetica', '', 8.5)
            self.set_text_color(71, 85, 105)
            self.cell(130, 5, v)
            self.ln(5)
        self.ln(4)

def generate_report():
    pdf = ProjectReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(12, 12, 12)

    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.ln(15)
    
    # Title Box
    pdf.set_fill_color(79, 70, 229)
    pdf.rect(12, 25, 186, 42, style='F')
    pdf.set_xy(16, 32)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "RetailPro AI - Machine Learning System", align='C')
    pdf.ln(11)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, "Complete Project Implementation Plan & Technical Reference Guide", align='C')
    
    pdf.ln(35)
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "ACADEMIC PROJECT TECHNICAL DOCUMENTATION", align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "For B.Tech / B.E. Computer Science & Engineering Project Report & Viva Voce", align='C')
    pdf.ln(15)

    # Meta Info Card
    meta_items = [
        ("Domain", "Applied Machine Learning, Retail Analytics & Supply Chain Management"),
        ("Primary Technology", "Python, Streamlit, Scikit-learn, Pandas, Plotly, Joblib"),
        ("Algorithms Implemented", "Linear Regression, Random Forest Regressor, K-Means Clustering"),
        ("Techniques Used", "Time-Series Lag Forecasting, RFM Segmentation, Safety Stock (ROP)"),
        ("Dataset", "Superstore Retail Dataset (9,994 Orders, 27 Cleaned Attributes)"),
        ("Target Users", "Store Managers, Inventory Controllers, Retail Merchandisers, CXOs"),
        ("Date of Publication", datetime.datetime.now().strftime("%B %Y"))
    ]
    pdf.key_value_box("PROJECT SPECIFICATION SUMMARY", meta_items)
    
    pdf.ln(8)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "DOCUMENT PURPOSE:")
    pdf.ln(6)
    pdf.body_text(
        "This document is a comprehensive, end-to-end technical reference manual designed to assist students and "
        "evaluators in understanding the full architectural design, mathematical models, feature engineering pipelines, "
        "and algorithmic implementations in RetailPro AI. It provides all data and outlines necessary to prepare a formal "
        "B.Tech Project Final Report, synopsis, and PowerPoint presentation (PPT) defense."
    )

    # -------------------------------------------------------------
    # CHAPTER 1: EXECUTIVE SUMMARY & PROBLEM DEFINITION
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(1, "Executive Summary & Problem Statement")
    
    pdf.section_title("1.1 Background & Motivation")
    pdf.body_text(
        "The retail industry generates vast quantities of transactional data daily. However, traditional retail management "
        "systems are largely descriptive and retrospective--they explain what happened in the past rather than predicting future "
        "consumer demand or advising when replenishment should occur. This lack of predictive intelligence leads to severe "
        "inefficiencies: global retailers lose over $1.1 Trillion annually to stockouts (out-of-stock items) and excess overstocking."
    )
    
    pdf.section_title("1.2 Problem Statement")
    pdf.body_text(
        "Modern retail businesses struggle with three fundamental operational challenges:\n"
        "1. Inaccurate Sales Forecasting: Inability to predict the revenue value of incoming order configurations due to non-linear "
        "interactions between order quantities, discounts, categories, and regional demographics.\n"
        "2. Stockouts and Overstocking: Lack of automated, mathematically formulated safety stock thresholds and reorder triggers, "
        "forcing warehouse buyers to guess replenishment dates.\n"
        "3. Generic Customer Engagement: Treating all shoppers identically rather than segmenting them by lifetime value, frequency, "
        "and recency of purchases."
    )

    pdf.section_title("1.3 Proposed Solution: RetailPro AI")
    pdf.body_text(
        "RetailPro AI transforms traditional retail management into an intelligent, data-driven system. Built using Python, Scikit-learn, "
        "and Streamlit, the application integrates supervised regression, chronological time-series demand forecasting, unsupervised "
        "K-Means clustering, and supply chain operations research formulas (Safety Stock & Reorder Point) into an interactive, real-time dashboard."
    )

    # -------------------------------------------------------------
    # CHAPTER 2: SYSTEM ARCHITECTURE & TECHNOLOGY STACK
    # -------------------------------------------------------------
    pdf.chapter_title(2, "System Architecture & Technology Stack")
    
    pdf.section_title("2.1 Technologies & Frameworks Used")
    tech_table = [
        ("Programming Language", "Python 3.10+ (Core backend and data science pipeline)"),
        ("Web Dashboard UI", "Streamlit (Modern reactive UI with custom CSS and instant navigation)"),
        ("Data Processing", "Pandas & NumPy (Vectorized dataframes, grouping, datetime operations)"),
        ("Machine Learning", "Scikit-Learn (Supervised regression, clustering, pipelines, metrics)"),
        ("Data Visualization", "Plotly Express & Plotly Graph Objects (Interactive 2D/3D charts)"),
        ("Model Serialization", "Joblib (High-efficiency binary serialization of trained models)"),
        ("PDF & Export", "FPDF2 (Automated technical documentation and report generation)"),
        ("Version Control", "Git & GitHub (Repository: github.com/pragathirajrev-lang/RetailPro)")
    ]
    pdf.key_value_box("TECHNOLOGY STACK ARCHITECTURE", tech_table)

    pdf.section_title("2.2 Modular Codebase Organization")
    pdf.bullet_point("app.py", "Main Streamlit entry point. Contains responsive UI layout, KPI card renderers, dynamic filters, high-speed cached loaders, and clean plain-English explanation boxes for all 9 dashboard sections.")
    pdf.bullet_point("ml_engine.py", "Machine learning core. Implements Scikit-learn pipelines (Linear Regression, Random Forest), chronological time-series forecasting, RFM feature engineering with K-Means clustering, and supply chain inventory formulas.")
    pdf.bullet_point("data_processing.py", "Data ingestion, cleaning, date manipulation, revenue aggregations, KPI calculations, and the rule-based AI Retail Assistant engine.")
    pdf.bullet_point("verify_system.py", "Standalone CLI test suite that validates dataset integrity, trains models, performs sample inference, and checks Joblib model persistence without opening a browser.")
    pdf.bullet_point("models/", "Directory holding serialized model files (forecast_model.joblib, kmeans_model.joblib, sales_models.joblib) enabling instantaneous app startup.")
    pdf.bullet_point("data/", "Contains clean Superstore transaction dataset (superstore.csv).")

    # -------------------------------------------------------------
    # CHAPTER 3: DATASET & PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(3, "Dataset & Feature Engineering Pipeline")

    pdf.section_title("3.1 Dataset Specification")
    pdf.body_text(
        "The project utilizes the benchmark Superstore Retail Dataset comprising 9,994 transaction records spanning four calendar years. "
        "After data hygiene verification, 9,983 records and 27 comprehensive attributes were retained."
    )
    
    ds_features = [
        ("Transactional Keys", "Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Product ID"),
        ("Customer Demographics", "Customer Name, Segment (Consumer, Corporate, Home Office), Country, City, State, Region"),
        ("Product Taxonomy", "Category (Furniture, Office Supplies, Technology), Sub-Category (17 distinct lines), Product Name"),
        ("Financial Metrics", "Sales ($), Quantity, Discount (0% to 80%), Profit ($), Profit Margin (%)"),
        ("Engineered Features", "Order_Year, Order_Month, Order_Day, Ship_Days, Unit_Price, Order_Month_Year")
    ]
    pdf.key_value_box("DATASET FEATURES & ATTRIBUTES", ds_features)

    pdf.section_title("3.2 Preprocessing & Feature Engineering Steps")
    pdf.bullet_point("Datetime Extraction", "Converted Order Date and Ship Date into datetime objects. Extracted Year, Month, Day, and calculated Ship_Days (Ship Date - Order Date) as an operational efficiency indicator.")
    pdf.bullet_point("Missing Value Imputation", "Verified that no critical financial columns contain NaN values. Handled edge cases by median imputation on numeric columns.")
    pdf.bullet_point("Outlier & Inconsistency Curation", "Corrected inconsistent string casing and stripped whitespace across Sub-Category and Region columns.")
    pdf.bullet_point("Categorical One-Hot Encoding", "Applied OneHotEncoder(handle_unknown='ignore') to transform high-cardinality nominal variables (Category, Sub-Category, Region, Segment) into sparse binary indicators.")
    pdf.bullet_point("Feature Scaling", "Standardized continuous numeric inputs (Quantity, Discount, Ship_Days) using StandardScaler to have zero mean and unit variance.")

    # -------------------------------------------------------------
    # CHAPTER 4: MACHINE LEARNING ALGORITHMS & MATHEMATICS
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(4, "Machine Learning Algorithms & Mathematical Models")

    pdf.section_title("4.1 Feature 1: Sales Price Prediction (Supervised Regression)")
    pdf.body_text(
        "Objective: Predict the exact dollar sales amount of an order based on its transactional parameters.\n\n"
        "Algorithms Compared:\n"
        "1. Multiple Linear Regression (Baseline Model): Assumes linear relationship y = w^T x + b. Fast, interpretable, but fails to capture non-linear pricing elasticities.\n"
        "2. Random Forest Regressor (Ensemble Model): Constructs an ensemble of 100 decorrelated decision trees using bootstrap aggregating (bagging). Effectively captures complex interaction effects between discounts, product tiers, and bulk quantities.\n\n"
        "Pipeline Architecture:\n"
        "Preprocessed via ColumnTransformer -> 80/20 Train-Test Split -> Scikit-learn Pipeline fitting -> Model evaluation -> Joblib serialization."
    )
    pdf.bullet_point("Evaluation Metrics", "R-squared (Coefficient of Determination), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).")
    pdf.bullet_point("Empirical Benchmark", "Random Forest achieved a lower Mean Absolute Error ($214.22) compared to Linear Regression ($219.06), successfully modeling non-linear discount thresholds.")

    pdf.section_title("4.2 Feature 2: Chronological Demand Forecasting (Time-Series)")
    pdf.body_text(
        "Objective: Forecast product demand volume (units) across monthly intervals up to 12 months in advance.\n\n"
        "Methodology & Data Leakage Prevention:\n"
        "Standard random cross-validation randomly mixes past and future records, introducing catastrophic look-ahead data leakage. "
        "To guarantee realistic real-world evaluation, our pipeline implements STRICT CHRONOLOGICAL TRAIN-TEST SPLITTING. The first 80% "
        "of time intervals form the training horizon, and the final 20% form the holdout test set.\n\n"
        "Engineered Time-Series Features:\n"
        "* Lag-1 Demand (Sales volume in month t-1)\n"
        "* Lag-2 Demand (Sales volume in month t-2)\n"
        "* Lag-3 Demand (Sales volume in month t-3)\n"
        "* Rolling 3-Month Mean Demand\n"
        "* Month Seasonality Indicator (captures November-December holiday retail spikes)\n\n"
        "Recursive Multi-Step Forecasting: Generates step t+1, appends it as the new lag feature, and iteratively predicts t+2 through t+12. "
        "Confidence intervals (95%) are computed via residual variance: Forecast +/- 1.96 * sigma_residual."
    )

    pdf.add_page()
    pdf.section_title("4.3 Feature 3: Customer Segmentation (Unsupervised K-Means)")
    pdf.body_text(
        "Objective: Cluster customer base into actionable behavioral cohorts without manual labeling.\n\n"
        "RFM Feature Engineering:\n"
        "* Recency (R): Number of days since the customer's last purchase.\n"
        "* Frequency (F): Total count of distinct orders placed by the customer.\n"
        "* Monetary (M): Total cumulative revenue ($) contributed by the customer.\n\n"
        "Skewness Correction: Retail monetary value is highly right-skewed (a few customers spend thousands, many spend small amounts). "
        "We apply a log-transformation: log1p(x) = ln(1 + x), followed by StandardScaler normalization.\n\n"
        "Optimal Cluster Selection (k=3):\n"
        "Evaluated via the Elbow Method (sum of squared distances) and confirmed by Silhouette Score analysis.\n\n"
        "Identified Customer Cohorts:\n"
        "1. High-Value VIPs: High monetary spend, high order frequency, low recency. Strategy: VIP loyalty rewards, early product access.\n"
        "2. Regular Customers: Moderate frequency and spend, steady recency. Strategy: Cross-selling and upselling campaigns.\n"
        "3. Low-Engagement / At-Risk: High recency (long inactivity), low spend. Strategy: Re-activation discounts and win-back emails."
    )

    pdf.section_title("4.4 Feature 4: Mathematical Inventory Optimization (Safety Stock & ROP)")
    pdf.body_text(
        "Objective: Calculate exact safety buffer stocks and reorder triggers for every product catalog item.\n\n"
        "Mathematical Formulations:\n"
        "1. Average Daily Demand (d_bar) = Total Units Sold / Active Sales Days\n"
        "2. Daily Demand Volatility (sigma_d) = Standard deviation of daily order quantities\n"
        "3. Safety Stock (SS) = Z * sigma_d * sqrt(L)\n"
        "   - Where L = Supplier Lead Time in days (user-adjustable, default 7 days)\n"
        "   - Where Z = Inverse standard normal cumulative distribution service factor (Z = 1.645 for 95% service level protection)\n"
        "4. Reorder Point (ROP) = (d_bar * L) + SS\n\n"
        "Replenishment Policy Logic:\n"
        "* If Current Stock <= 0.5 * SS: Flag as 'Critical (Stockout Risk)'\n"
        "* If Current Stock <= ROP: Flag as 'Reorder Required' -> Recommended Order Qty = (ROP * 2) - Current Stock\n"
        "* If Current Stock > 3 * ROP: Flag as 'Overstocked (Capital Risk)'\n"
        "* Else: Flag as 'Healthy Stock'"
    )

    # -------------------------------------------------------------
    # CHAPTER 5: DASHBOARD MODULES & USER INTERFACE
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(5, "Dashboard Modules & User Interface Design")
    
    pdf.body_text(
        "RetailPro AI is organized into 9 unified dashboard views accessible via the sidebar navigation. "
        "Every view is backed by instant-load memory caching and provides a dedicated 'Overview Card' detailing the dashboard purpose and contents."
    )

    modules_summary = [
        ("1. Executive Dashboard", "High-level summary of store revenue ($2.3M), net profit, total orders, monthly trend line, and category breakdown pie chart."),
        ("2. Sales Analytics", "Detailed exploration of monthly sales trajectories, revenue rankings across 17 sub-categories, and 4-quadrant regional profitability tables."),
        ("3. ML Sales Prediction", "Interactive calculator allowing users to select order variables and receive instant sales price predictions with error metrics."),
        ("4. Demand Forecasting", "Interactive time-series forecast displaying historical vs future unit demand for 3, 6, or 12 months with shaded 95% confidence bands."),
        ("5. Customer Segmentation", "Visualizes RFM cohorts using interactive Plotly charts, cohort summary metrics, and a searchable customer lookup tool."),
        ("6. Inventory Optimization", "Interactive policy sliders for Lead Time and Service Level that compute real-time Safety Stock, ROP, and purchase orders."),
        ("7. Product Performance", "Identifies top revenue moneymakers vs bottom loss-making products to guide inventory clearance and discontinuation."),
        ("8. ML Model Evaluation", "Displays formal academic metrics: R², MAE, RMSE, error comparison bar charts, and clustering Silhouette Score."),
        ("9. AI Retail Assistant", "Conversational interface answering store analytics queries (restocking needs, top sellers, revenue) in natural language.")
    ]
    pdf.key_value_box("DASHBOARD MODULE ARCHITECTURE", modules_summary)

    # -------------------------------------------------------------
    # CHAPTER 6: PPT PRESENTATION SLIDE-BY-SLIDE GUIDE
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(6, "PowerPoint (PPT) Presentation Slide-by-Slide Outline")
    pdf.body_text("Use this 12-slide template to prepare your official academic presentation:")

    slides = [
        ("Slide 1: Title Slide", "Project Title: RetailPro AI - Machine Learning-Based Retail & Inventory Management System. Student name, roll number, department (CSE), college name, and guide name."),
        ("Slide 2: Problem Statement", "Highlight traditional retail challenges: stockouts ($1.1T loss), overstock holding costs, inaccurate forecasting, and unsegmented customer marketing."),
        ("Slide 3: Project Objectives", "Enumerate key goals: ML sales price prediction, chronological demand forecasting, K-Means customer segmentation, mathematical safety stock, and interactive dashboards."),
        ("Slide 4: System Architecture", "Diagram showing Data Ingestion (CSV) -> Data Cleaning (Pandas) -> Scikit-learn ML Pipelines -> Joblib Serialization -> Streamlit Interactive UI."),
        ("Slide 5: Dataset & Preprocessing", "Superstore dataset overview: 9,983 records, 27 features, datetime parsing, One-Hot Encoding, StandardScaler, and log-transformation."),
        ("Slide 6: ML Technique 1 - Sales Prediction", "Explain Linear Regression vs. Random Forest Regressor. Present R², MAE ($214), RMSE, and show why ensemble trees outperform linear models on retail pricing."),
        ("Slide 7: ML Technique 2 - Demand Forecasting", "Explain chronological splitting to prevent data leakage. Detail Lag features, seasonality indicators, and multi-step recursive forecasting with confidence bands."),
        ("Slide 8: ML Technique 3 - Customer Segmentation", "Detail RFM feature engineering (Recency, Frequency, Monetary). Explain log-transform and present the 3 identified clusters (High-Value, Regular, At-Risk)."),
        ("Slide 9: Feature 4 - Inventory Optimization", "Present the mathematical formulas: Safety Stock = Z * sigma * sqrt(L), and Reorder Point = (Daily Demand * L) + SS. Explain replenishment alerts."),
        ("Slide 10: Model Evaluation Benchmarks", "Summary slide comparing all models with formal metrics: Regression MAE, Forecasting R², and K-Means Silhouette Score."),
        ("Slide 11: Live Demonstration Screenshots", "Include screenshots of the Executive Dashboard, Sales Prediction Tool, Demand Forecasting Chart, and Customer Clusters."),
        ("Slide 12: Conclusion & Future Scope", "Key takeaways: proactive supply chain automation. Future scope: ERP database integration, deep learning LSTM forecasting, automated supplier email purchase orders.")
    ]

    for title, desc in slides:
        pdf.bullet_point(title, desc)

    # -------------------------------------------------------------
    # CHAPTER 7: VIVA VOCE DEFENSE QUESTIONS & ANSWERS
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.chapter_title(7, "Viva Voce / Oral Defense Questions & Answers")
    pdf.body_text("Frequently asked questions by academic project examiners with recommended answers:")

    qas = [
        ("Q1: Why did you choose Random Forest over Linear Regression for Sales Prediction?",
         "Linear Regression assumes a straight-line relationship between inputs and output. In retail, discounts and quantities have non-linear price effects (e.g. 50% off on electronics behaves differently than on office pens). Random Forest builds 100 decision trees that split on combinations of features, yielding a lower Mean Absolute Error ($214 vs $219)."),
        
        ("Q2: What is 'Data Leakage' in time-series forecasting, and how did you prevent it?",
         "Data leakage occurs when information from the future leaks into the training set. If we used standard random train-test splitting, the model would train on December data and test on November data, which is impossible in the real world. We prevented this by using strict CHRONOLOGICAL splitting: the first 80% of months for training, and the final 20% for testing."),

        ("Q3: Why did you apply log-transformation prior to K-Means clustering?",
         "In retail, monetary spend and frequency are heavily right-skewed--a few customers spend thousands of dollars while the vast majority spend small amounts. Without log-transformation, K-Means clustering is distorted by extreme outliers. Applying log1p(x) compresses the scale into a bell-shaped distribution, allowing Euclidean distance to cluster customers fairly."),

        ("Q4: How is Safety Stock mathematically determined?",
         "Safety Stock is formulated as SS = Z * sigma_d * sqrt(L). Sigma_d represents daily demand standard deviation (demand volatility), L is supplier delivery lead time, and Z is the z-score corresponding to desired service level (1.645 for 95% protection against stockouts during supplier transit)."),

        ("Q5: How does the application achieve instant page loading?",
         "We pre-trained and serialized all Scikit-learn pipelines into binary Joblib artifacts in the /models directory. At runtime, Streamlit caches models and aggregations in RAM using @st.cache_resource and @st.cache_data, eliminating repeated fitting and providing sub-100ms transitions.")
    ]

    for q, a in qas:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(79, 70, 229)
        pdf.multi_cell(0, 5.5, q)
        pdf.ln(1)
        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 5.2, a)
        pdf.ln(4)

    # Save to file
    output_filename = "RetailPro_Implementation_Plan_and_Project_Report.pdf"
    pdf.output(output_filename)
    print(f"Successfully generated: {output_filename}")

if __name__ == "__main__":
    generate_report()
