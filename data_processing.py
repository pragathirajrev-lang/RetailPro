import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
    df = df.dropna()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df = df.drop_duplicates()
    df['Profit Margin (%)'] = (df['Profit'] / df['Sales']) * 100
    df['Month-Year'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    df['Quarter'] = df['Order Date'].dt.quarter
    df['DayOfWeek'] = df['Order Date'].dt.day_name()
    return df

def filter_data(df, date_range, categories, products, regions, segments):
    mask = pd.Series(True, index=df.index)
    if date_range and len(date_range) == 2:
        mask &= (df['Order Date'] >= pd.Timestamp(date_range[0])) & (df['Order Date'] <= pd.Timestamp(date_range[1]))
    if categories:
        mask &= df['Category'].isin(categories)
    if products:
        mask &= df['Product Name'].isin(products)
    if regions:
        mask &= df['Region'].isin(regions)
    if segments:
        mask &= df['Segment'].isin(segments)
    return df[mask].copy()

def get_executive_kpis(df):
    total_revenue = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = df['Order ID'].nunique()
    total_products_sold = df['Quantity'].sum()

    inv_value = df.groupby('Product Name')['Quantity'].sum()
    inv_value_total = (inv_value * df.groupby('Product Name')['Sales'].sum() / df.groupby('Product Name')['Quantity'].sum().replace(0, 1)).sum()

    turnover = total_products_sold / max(len(inv_value), 1)

    top_product = df.groupby('Product Name')['Quantity'].sum().idxmax() if not df.empty else 'N/A'

    low_stock_count = len(df.groupby('Product Name')['Quantity'].sum()[df.groupby('Product Name')['Quantity'].sum() < 10])

    dates = df['Order Date'].sort_values()
    mid_point = dates.iloc[len(dates)//2] if len(dates) > 1 else dates.iloc[0]
    prev_period = df[df['Order Date'] < mid_point]
    curr_period = df[df['Order Date'] >= mid_point]

    def growth(curr, prev):
        if prev == 0:
            return 0.0
        return float(((curr - prev) / prev) * 100)

    rev_growth = growth(curr_period['Sales'].sum(), prev_period['Sales'].sum())
    profit_growth = growth(curr_period['Profit'].sum(), prev_period['Profit'].sum())
    order_growth = growth(curr_period['Order ID'].nunique(), prev_period['Order ID'].nunique())

    return {
        'Total Revenue': (total_revenue, rev_growth),
        'Total Profit': (total_profit, profit_growth),
        'Total Orders': (total_orders, order_growth),
        'Total Products Sold': (total_products_sold, 0.0),
        'Inventory Value': (inv_value_total, 0.0),
        'Inventory Turnover Rate': (turnover, 0.0),
        'Best Selling Product': (top_product, 0.0),
        'Low Stock Products Count': (low_stock_count, 0.0)
    }

def get_monthly_sales_trend(df):
    monthly = df.groupby('Month-Year').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index()
    monthly = monthly.sort_values('Month-Year')
    monthly['Month-Year'] = monthly['Month-Year'].astype(str)
    return monthly

def get_category_revenue(df):
    return df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Sales', ascending=False)

def get_sub_category_revenue(df):
    if 'Sub-Category' in df.columns:
        return df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Sales', ascending=False)
    return pd.DataFrame(columns=['Category', 'Sub-Category', 'Sales', 'Profit', 'Quantity'])


def get_product_revenue(df):
    return df.groupby('Product Name').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Sales', ascending=False)

def get_top_products(df, n=10):
    return get_product_revenue(df).head(n)

def get_sales_distribution(df):
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000, 50000]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
    df_copy = df.copy()
    df_copy['Sales Range'] = pd.cut(df_copy['Sales'], bins=bins, labels=labels)
    return df_copy.groupby('Sales Range').agg({'Sales': 'sum', 'Order ID': 'nunique'}).reset_index()

def get_region_performance(df):
    return df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Order ID': 'nunique'}).reset_index().sort_values('Sales', ascending=False)

def get_state_performance(df):
    return df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Sales', ascending=False)

def get_profit_analysis(df):
    return df.groupby('Category').agg({'Profit': 'sum', 'Sales': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Profit', ascending=False)

def get_inventory_summary(df):
    inv = df.groupby('Product Name').agg({
        'Quantity': 'sum',
        'Sales': 'sum',
        'Profit': 'sum',
        'Category': 'first',
        'Sub-Category': 'first'
    }).reset_index()
    inv['Avg Price'] = inv['Sales'] / inv['Quantity'].replace(0, 1)
    inv['Inventory Value'] = inv['Quantity'] * inv['Avg Price']
    return inv

def get_inventory_status(df):
    inv = get_inventory_summary(df)
    qty = inv['Quantity']
    q75 = qty.quantile(0.75)
    inv['Turnover'] = df.groupby('Product Name')['Quantity'].sum().reindex(inv['Product Name']).values / inv['Quantity'].replace(0, 1).values
    inv['Status'] = inv['Quantity'].apply(
        lambda x: 'Critical Stock' if x < 5 else ('Low Stock' if x < 15 else ('Healthy Stock' if x <= q75 else 'Overstocked'))
    )
    inv['Status Color'] = inv['Status'].map({
        'Critical Stock': 'red',
        'Low Stock': 'orange',
        'Healthy Stock': 'green',
        'Overstocked': 'blue'
    })
    return inv.sort_values('Quantity')

def get_stock_aging(df):
    inv = get_inventory_summary(df)
    latest = df['Order Date'].max()
    last_sale = df.groupby('Product Name')['Order Date'].max().reset_index()
    last_sale.columns = ['Product Name', 'Last Sale Date']
    inv = inv.merge(last_sale, on='Product Name', how='left')
    inv['Days Since Last Sale'] = (latest - inv['Last Sale Date']).dt.days.fillna(999)
    inv['Aging Category'] = inv['Days Since Last Sale'].apply(
        lambda x: 'Fast Moving' if x < 30 else ('Normal' if x < 90 else ('Slow Moving' if x < 180 else 'Dead Stock'))
    )
    return inv

def get_overstocked_products(df):
    inv = get_inventory_status(df)
    return inv[inv['Status'] == 'Overstocked'].sort_values('Quantity', ascending=False)

def get_understocked_products(df):
    inv = get_inventory_status(df)
    return inv[inv['Status'].isin(['Critical Stock', 'Low Stock'])].sort_values('Quantity')

def get_inventory_alerts(df):
    inv = get_inventory_status(df)
    alerts = []
    for _, row in inv.iterrows():
        if row['Status'] == 'Critical Stock':
            alerts.append(f"CRITICAL: {row['Product Name']} stock ({int(row['Quantity'])}) is critically low. Immediate restocking required!")
        elif row['Status'] == 'Low Stock':
            alerts.append(f"WARNING: {row['Product Name']} stock ({int(row['Quantity'])}) is below reorder threshold. Reorder soon.")
        elif row['Status'] == 'Overstocked':
            alerts.append(f"INFO: {row['Product Name']} inventory ({int(row['Quantity'])}) exceeds expected demand. Consider promotion.")
    return alerts

def get_forecast(df, periods=3):
    monthly = df.groupby('Month-Year').agg({'Sales': 'sum'}).reset_index().sort_values('Month-Year')
    monthly['MonthNum'] = range(len(monthly))
    monthly['Sales'] = monthly['Sales'].astype(float)

    if len(monthly) < 3:
        return monthly, pd.DataFrame(), None

    X = monthly['MonthNum'].values.reshape(-1, 1)
    y = monthly['Sales'].values
    model = LinearRegression()
    model.fit(X, y)

    future_nums = np.arange(len(monthly), len(monthly) + periods).reshape(-1, 1)
    future_preds = model.predict(future_nums)
    residuals = y - model.predict(X)
    std_err = np.std(residuals)

    last_date = pd.Period(monthly['Month-Year'].iloc[-1], freq='M')
    future_dates = [str(last_date + i) for i in range(1, periods + 1)]

    future_df = pd.DataFrame({
        'Month-Year': future_dates,
        'Forecast': future_preds,
        'Lower CI': future_preds - 1.96 * std_err,
        'Upper CI': future_preds + 1.96 * std_err
    })

    combined = pd.concat([
        monthly[['Month-Year', 'Sales']].rename(columns={'Sales': 'Value'}),
        pd.DataFrame({'Month-Year': future_dates, 'Value': future_preds})
    ])
    combined['Type'] = ['Actual'] * len(monthly) + ['Forecast'] * periods

    recommendations = []
    for i, (_, row) in enumerate(future_df.iterrows()):
        if i < len(monthly):
            last_actual = monthly['Sales'].iloc[-1]
        else:
            last_actual = future_df['Forecast'].iloc[i-1]
        growth_pct = ((row['Forecast'] - last_actual) / last_actual) * 100
        if growth_pct > 10:
            recommendations.append(f"Increase inventory by {int(growth_pct)}% next month due to expected demand growth of {growth_pct:.1f}%.")
        elif growth_pct < -10:
            recommendations.append(f"Reduce inventory by {int(abs(growth_pct))}% next month due to expected demand decrease of {abs(growth_pct):.1f}%.")

    return combined, future_df, recommendations

def get_customer_segments(df):
    customer = df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    customer.columns = ['Customer Name', 'Total Sales', 'Total Profit', 'Order Count', 'Total Quantity']
    customer['Avg Order Value'] = customer['Total Sales'] / customer['Order Count'].replace(0, 1)
    customer['Purchase Frequency'] = customer['Order Count'] / max(df['Order Date'].dt.year.nunique(), 1)
    customer['CLV'] = customer['Total Sales'] * 0.3

    sales_q75 = customer['Total Sales'].quantile(0.75)
    sales_q50 = customer['Total Sales'].quantile(0.50)
    freq_median = customer['Order Count'].median()

    def segment(row):
        if row['Total Sales'] >= sales_q75 and row['Order Count'] >= freq_median * 1.5:
            return 'High Value Customers'
        elif row['Order Count'] >= freq_median * 2:
            return 'Loyal Customers'
        elif row['Total Sales'] >= sales_q50:
            return 'Occasional Customers'
        else:
            return 'At-Risk Customers'

    customer['Segment'] = customer.apply(segment, axis=1)
    customer['Days Since Last Purchase'] = np.random.randint(1, 180, size=len(customer))
    return customer

def get_customer_recommendations(df):
    segments = get_customer_segments(df)
    recs = []
    high_val = segments[segments['Segment'] == 'High Value Customers']
    loyal = segments[segments['Segment'] == 'Loyal Customers']
    at_risk = segments[segments['Segment'] == 'At-Risk Customers']
    occasional = segments[segments['Segment'] == 'Occasional Customers']

    recs.append(f"Send VIP rewards to {len(high_val)} high-value customers to maintain loyalty.")
    recs.append(f"Offer loyalty program to {len(loyal)} loyal customers with exclusive discounts.")
    recs.append(f"Re-engage {len(at_risk)} at-risk customers with targeted campaigns and special offers.")
    recs.append(f"Convert {len(occasional)} occasional customers with personalized product recommendations.")
    return recs

def get_business_insights(df):
    insights = []
    cat_rev = get_category_revenue(df)
    if not cat_rev.empty:
        insights.append(f"Highest Revenue Category: {cat_rev.iloc[0]['Category']} (${cat_rev.iloc[0]['Sales']:,.2f})")
        insights.append(f"Lowest Performing Category: {cat_rev.iloc[-1]['Category']} (${cat_rev.iloc[-1]['Sales']:,.2f})")

    prod = get_product_revenue(df)
    if not prod.empty:
        top = prod.iloc[0]
        insights.append(f"Most Profitable Product: {top['Product Name']} (${top['Profit']:,.2f})")

    inv = get_inventory_status(df)
    critical = len(inv[inv['Status'] == 'Critical Stock'])
    overstocked = len(inv[inv['Status'] == 'Overstocked'])
    insights.append(f"Inventory Risk Areas: {critical} critically low items, {overstocked} overstocked items")

    monthly = get_monthly_sales_trend(df)
    if len(monthly) > 1:
        peak_month = monthly.loc[monthly['Sales'].idxmax(), 'Month-Year']
        low_month = monthly.loc[monthly['Sales'].idxmin(), 'Month-Year']
        insights.append(f"Peak Sales Month: {peak_month}")
        insights.append(f"Lowest Sales Month: {low_month}")

    region = get_region_performance(df)
    if not region.empty:
        insights.append(f"Top Region: {region.iloc[0]['Region']} (${region.iloc[0]['Sales']:,.2f})")
        insights.append(f"Region {region.iloc[-1]['Region']} needs attention with only ${region.iloc[-1]['Sales']:,.2f} in sales")

    margin = df['Profit Margin (%)'].mean()
    insights.append(f"Average Profit Margin: {margin:.1f}%")
    return insights

def get_seasonal_patterns(df):
    df_copy = df.copy()
    df_copy['Month'] = df_copy['Order Date'].dt.month_name()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    seasonal = df_copy.groupby('Month').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reindex(months_order).reset_index()
    return seasonal

def get_ai_response(df, question):
    q = question.lower()

    if 'restock' in q or 'low stock' in q or 'reorder' in q:
        alerts = get_inventory_alerts(df)
        critical = [a for a in alerts if a.startswith('CRITICAL')]
        warnings_list = [a for a in alerts if a.startswith('WARNING')]
        if not critical and not warnings_list:
            return "No products need immediate restocking. All inventory levels are healthy."
        response = "**Inventory Restock Report:**\n\n"
        if critical:
            response += "**Critical (Immediate Action Required):**\n"
            for a in critical[:5]:
                response += f"- {a}\n"
            response += "\n"
        if warnings_list:
            response += "**Warning (Restock Soon):**\n"
            for a in warnings_list[:5]:
                response += f"- {a}\n"
            return response

    if 'top selling' in q or 'best selling' in q or 'top product' in q:
        top = get_top_products(df, 10)
        response = "**Top 10 Best Selling Products:**\n\n"
        for i, (_, row) in enumerate(top.iterrows(), 1):
            response += f"{i}. {row['Product Name']} - ${row['Sales']:,.2f} Revenue, {int(row['Quantity'])} units sold\n"
        return response

    if 'profit' in q and ('category' in q or 'highest' in q):
        prof = get_profit_analysis(df)
        response = "**Profit Analysis by Category:**\n\n"
        for _, row in prof.iterrows():
            response += f"- {row['Category']}: ${row['Profit']:,.2f} profit on ${row['Sales']:,.2f} sales\n"
        response += f"\nMost profitable category: **{prof.iloc[0]['Category']}**"
        return response

    if 'predict' in q or 'forecast' in q or 'next month' in q:
        combined, future_df, recs = get_forecast(df)
        if future_df.empty:
            return "Not enough data to make predictions. Need at least 3 months of data."
        response = "**Sales Forecast:**\n\n"
        for _, row in future_df.iterrows():
            response += f"- {row['Month-Year']}: Predicted ${row['Forecast']:,.2f} "
            response += f"(CI: ${row['Lower CI']:,.2f} - ${row['Upper CI']:,.2f})\n"
        response += "\n**Next Month Prediction:**\n"
        next_month = future_df.iloc[0]
        response += f"Expected Sales: **${next_month['Forecast']:,.2f}**\n"
        if recs:
            response += f"\n**Recommendation:** {recs[0]}"
        return response

    if 'inventory risk' in q or 'risk' in q:
        inv = get_inventory_status(df)
        critical = inv[inv['Status'] == 'Critical Stock']
        overstocked = inv[inv['Status'] == 'Overstocked']
        response = "**Inventory Risk Assessment:**\n\n"
        response += f"**Critical Stock Items ({len(critical)}):**\n"
        for _, row in critical.head(5).iterrows():
            response += f"- {row['Product Name']}: Only {int(row['Quantity'])} units left\n"
        response += f"\n**Overstocked Items ({len(overstocked)}):**\n"
        for _, row in overstocked.head(5).iterrows():
            response += f"- {row['Product Name']}: {int(row['Quantity'])} units (excess stock)\n"
        if len(critical) == 0 and len(overstocked) == 0:
            response += "No significant inventory risks detected."
        return response

    if 'revenue' in q or 'sales' in q:
        kpis = get_executive_kpis(df)
        response = "**Sales & Revenue Summary:**\n\n"
        response += f"Total Revenue: ${kpis['Total Revenue'][0]:,.2f}\n"
        response += f"Total Orders: {kpis['Total Orders'][0]:,}\n"
        response += f"Total Products Sold: {kpis['Total Products Sold'][0]:,}\n"
        response += f"Best Selling Product: {kpis['Best Selling Product'][0]}\n"
        return response

    if 'category' in q or 'categories' in q:
        cat = get_category_revenue(df)
        response = "**Category Performance:**\n\n"
        for _, row in cat.iterrows():
            response += f"- {row['Category']}: ${row['Sales']:,.2f} Revenue, ${row['Profit']:,.2f} Profit\n"
        return response

    if 'region' in q:
        region = get_region_performance(df)
        response = "**Regional Performance:**\n\n"
        for _, row in region.iterrows():
            response += f"- {row['Region']}: ${row['Sales']:,.2f} Revenue, {int(row['Order ID'])} Orders\n"
        return response

    if 'customer' in q:
        segs = get_customer_segments(df)
        counts = segs['Segment'].value_counts()
        response = "**Customer Segmentation Summary:**\n\n"
        for seg, count in counts.items():
            avg_sales = segs[segs['Segment'] == seg]['Total Sales'].mean()
            response += f"- {seg}: {count} customers (Avg ${avg_sales:,.2f} per customer)\n"
        return response

    return ("I can help you with:\n\n"
            "- **Restock alerts**: 'Which products need restocking?'\n"
            "- **Top products**: 'What are the top selling products?'\n"
            "- **Profit analysis**: 'Which category generated highest profit?'\n"
            "- **Sales forecast**: 'Predict next month sales'\n"
            "- **Inventory risks**: 'Show inventory risks'\n"
            "- **Revenue summary**: 'Show sales summary'\n"
            "- **Category analysis**: 'Show category performance'\n"
            "- **Regional data**: 'Show regional performance'\n"
            "- **Customer insights**: 'Show customer segments'")

def get_ceo_report(df):
    kpis = get_executive_kpis(df)
    insights = get_business_insights(df)
    alerts = get_inventory_alerts(df)
    combined, future_df, recs = get_forecast(df)

    revenue_growth = kpis['Total Revenue'][1]
    profit_growth = kpis['Total Profit'][1]
    order_growth = kpis['Total Orders'][1]

    report = []

    report.append("RETAIL ANALYTICS - CEO EXECUTIVE REPORT")
    report.append(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    report.append("=" * 60)

    report.append("\n1. EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Revenue: ${kpis['Total Revenue'][0]:,.2f} ({revenue_growth:+.1f}% vs previous period)")
    report.append(f"Total Profit: ${kpis['Total Profit'][0]:,.2f} ({profit_growth:+.1f}% vs previous period)")
    report.append(f"Total Orders: {kpis['Total Orders'][0]:,} ({order_growth:+.1f}% vs previous period)")
    report.append(f"Total Products Sold: {kpis['Total Products Sold'][0]:,}")
    report.append(f"Inventory Value: ${kpis['Inventory Value'][0]:,.2f}")
    report.append(f"Inventory Turnover Rate: {kpis['Inventory Turnover Rate'][0]:.2f}")
    report.append(f"Best Selling Product: {kpis['Best Selling Product'][0]}")
    report.append(f"Low Stock Products: {kpis['Low Stock Products Count'][0]}")

    report.append("\n2. KEY INSIGHTS")
    report.append("-" * 40)
    report.append(f"Revenue Growth: {revenue_growth:+.1f}%")
    if revenue_growth > 0:
        report.append("POSITIVE: Revenue is growing. Business expansion on track.")
    else:
        report.append("ATTENTION: Revenue is declining. Strategic review recommended.")
    report.append(f"Profit Growth: {profit_growth:+.1f}%")
    if profit_growth > 0:
        report.append("POSITIVE: Profit margins are improving.")
    else:
        report.append("ATTENTION: Profit margins need improvement.")

    cat_rev = get_category_revenue(df)
    if not cat_rev.empty:
        top_cat = cat_rev.iloc[0]
        report.append(f"\nTop Category: {top_cat['Category']} contributes {top_cat['Sales']/df['Sales'].sum()*100:.1f}% of sales.")

    report.append("\n3. INVENTORY STATUS")
    report.append("-" * 40)
    critical_count = sum(1 for a in alerts if a.startswith('CRITICAL'))
    warning_count = sum(1 for a in alerts if a.startswith('WARNING'))
    info_count = sum(1 for a in alerts if a.startswith('INFO'))
    report.append(f"Critical Items: {critical_count} (immediate action needed)")
    report.append(f"Low Stock Items: {warning_count} (restock soon)")
    report.append(f"Overstocked Items: {info_count} (excess inventory)")
    report.append(f"Total Inventory Alerts: {len(alerts)}")

    if not future_df.empty:
        report.append(f"\n4. FORECAST & RECOMMENDATIONS")
        report.append("-" * 40)
        next_pred = future_df.iloc[0]
        report.append(f"Expected Sales Next Month: ${next_pred['Forecast']:,.2f}")
        report.append(f"Confidence Interval: ${next_pred['Lower CI']:,.2f} - ${next_pred['Upper CI']:,.2f}")
        if len(monthly_sales := get_monthly_sales_trend(df)) > 0:
            prev = monthly_sales['Sales'].iloc[-1] if len(monthly_sales) > 0 else 0
            expected_growth = ((next_pred['Forecast'] - prev) / prev) * 100 if prev > 0 else 0
            report.append(f"Expected Growth: {expected_growth:+.1f}%")
        if recs:
            report.append(f"\nRecommendations:")
            for r in recs:
                report.append(f"  - {r}")

    report.append(f"\n5. RECOMMENDED ACTIONS")
    report.append("-" * 40)
    for insight in insights:
        report.append(f"  - {insight}")

    if critical_count > 0:
        report.append(f"  - Immediately restock {critical_count} critically low products to prevent stock-outs.")
    if profit_growth < 0:
        report.append("  - Review pricing strategy and cost structure to improve profitability.")
    report.append("  - Focus marketing on top-performing categories and regions.")
    report.append("  - Implement inventory optimization for overstocked items.")

    report.append("\n" + "=" * 60)
    report.append("END OF CEO REPORT")
    return '\n'.join(report)
