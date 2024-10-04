
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset for E-commerce sales
data = {
    'InvoiceDate': pd.date_range(start='2022-01-01', periods=24, freq='M'),
    'Quantity': [500, 600, 450, 700, 550, 630, 720, 800, 900, 850, 750, 900, 950, 990, 1020, 800, 600, 550, 700, 850, 940, 1000, 900, 980],
    'UnitPrice': [10, 12, 11, 9, 10, 11, 12, 9, 8, 10, 11, 12, 13, 10, 9, 10, 11, 9, 8, 9, 10, 12, 11, 10],
    'ProductCategory': ['Electronics', 'Furniture', 'Clothing', 'Electronics', 'Furniture', 'Clothing', 
                        'Electronics', 'Furniture', 'Clothing', 'Electronics', 'Furniture', 'Clothing',
                        'Electronics', 'Furniture', 'Clothing', 'Electronics', 'Furniture', 'Clothing',
                        'Electronics', 'Furniture', 'Clothing', 'Electronics', 'Furniture', 'Clothing']
}

df = pd.DataFrame(data)

# Creating 'Month' column and grouping data by month for total sales
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.to_period('M')
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
df_grouped = df.groupby('Month').agg({'TotalSales': 'sum'}).reset_index()
df_grouped['Month'] = df_grouped['Month'].dt.to_timestamp()

# Step 1: Visualizing the monthly sales trend
plt.figure(figsize=(10,6))
plt.plot(df_grouped['Month'], df_grouped['TotalSales'], marker='o', linestyle='-')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Step 2: Visualizing sales by product category
product_sales = df.groupby('ProductCategory')['Quantity'].sum()

plt.figure(figsize=(8,5))
product_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# Step 3: Preparing data for machine learning - Linear Regression
X = df_grouped.index.values.reshape(-1, 1)
y = df_grouped['TotalSales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Step 4: Visualizing real sales vs predicted sales
plt.figure(figsize=(10,6))
plt.plot(df_grouped['Month'], df_grouped['TotalSales'], label='Real Sales', marker='o')
plt.plot(df_grouped['Month'].iloc[X_test.flatten()], y_pred, label='Predicted Sales', marker='x', linestyle='--')
plt.title('Real vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Error calculation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
