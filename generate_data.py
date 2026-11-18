import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_simple_data():
    # Setup
    np.random.seed(42)
    products = {
        'Electronics': ['iPhone 15', 'Samsung S23', 'MacBook Air', 'Dell XPS', 'Sony Headphones'],
        'Furniture': ['Gaming Chair', 'Office Desk', 'Bookshelf', 'Sofa', 'Dining Table'],
        'Appliances': ['Air Conditioner', 'Heater', 'Microwave', 'Refrigerator', 'Washing Machine']
    }
    customers = ['Pragathi', 'Raj', 'Alex', 'Sarah', 'John', 'Emma', 'David', 'Sophia', 'James', 'Olivia']
    sub_categories = {
        'Electronics': ['Phones', 'Laptops', 'Audio', 'Accessories', 'Tablets'],
        'Furniture': ['Chairs', 'Desks', 'Storage', 'Seating', 'Tables'],
        'Appliances': ['Climate Control', 'Kitchen', 'Laundry', 'Cooking', 'Cleaning']
    }
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    # Generate 2000 rows
    for _ in range(2000):
        cat = np.random.choice(list(products.keys()))
        sub_cat = np.random.choice(sub_categories[cat])
        prod = np.random.choice(products[cat])
        reg = np.random.choice(regions)
        cust = np.random.choice(customers)
        date = start_date + timedelta(days=np.random.randint(0, 480))
        
        # Logic for seasonal sales
        month = date.month
        # RANDOMIZE QUANTITY to ensure some are low
        base_qty = np.random.randint(1, 25) 
        
        # Boost AC in Summer
        if prod == 'Air Conditioner' and month in [5, 6, 7]:
            base_qty += 10
            
        qty = base_qty
        price = np.random.randint(50, 1500)
        sales = qty * price
        profit = sales * np.random.uniform(0.1, 0.3)
        
        # Fixed State Mapping for Heatmap
        state_map = {'North': 'NY', 'South': 'TX', 'East': 'FL', 'West': 'CA'}
        st_code = state_map[reg]
        
        data.append([
            f"ORD-{np.random.randint(1000, 9999)}", date, cust, reg, cat, sub_cat, prod, qty, sales, profit, st_code
        ])
    
    df = pd.DataFrame(data, columns=['Order ID', 'Order Date', 'Customer Name', 'Region', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Sales', 'Profit', 'State'])
    df.to_csv('data/simple_retail_data.csv', index=False)
    print("Simple dataset generated successfully!")

if __name__ == "__main__":
    generate_simple_data()
