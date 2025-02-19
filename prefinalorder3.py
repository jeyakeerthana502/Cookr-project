import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from operator import itemgetter

# Define training data based on rules
rules_data = {
    'Order_Source': ['Same Kitchen', 'Different Kitchens', 'Same Kitchen', 'Different Kitchens', 'Different Kitchens', 'Same Kitchen', 'Different Kitchens', 'Same Kitchen'],
    'Order_Destination': ['Same Customer', 'Same Customer', 'Different Customers', 'Same Customer', 'Same Customer', 'Same Customer', 'Different Customers', 'Different Customers'],
    'Distance_Between_Kitchens': [0, 1, 0, 1, 1, 0, 1, 0],
    'Ready_Time_Difference': [10, 10, 10, 10, 10, 10, 10, 10],
    'Pickup_Strategy': ['Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider'],
    'Rule': [1, 2, 3, 4, 5, 6, 7, 8]
}

# Create a DataFrame from rules_data
df_rules = pd.DataFrame(rules_data)

# Define features and target variable
features = ['Order_Source', 'Order_Destination', 'Distance_Between_Kitchens', 'Ready_Time_Difference', 'Pickup_Strategy']
target = 'Rule'

# Split dataset into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(df_rules[features], df_rules[target], test_size=0.2, random_state=42)

# Define transformer for categorical features with OneHotEncoder (handling unknown categories)
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Order_Source', 'Order_Destination', 'Pickup_Strategy'])],
    remainder='passthrough'
)

# Create a pipeline with preprocessor and logistic regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Fit model to training data
model.fit(X_train, y_train)

# Sample orders for prediction
orders = [
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 0, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider'},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider'},
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider'},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider'},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 2, 'Ready_Time_Difference': 15, 'Pickup_Strategy': 'Same Rider'},
]

# Convert orders to DataFrame for prediction
df_orders = pd.DataFrame(orders)

# Predict batch assignment rules
df_orders['assigned_rule'] = model.predict(df_orders)

# Group orders into batches based on assigned_rule
grouped_orders = df_orders.groupby('assigned_rule')

# Print order batches
for rule, batch_orders in grouped_orders:
    print(f"\nBatch ID: {rule}")
    for _, order in batch_orders.iterrows():
        print(f"Order Source: {order['Order_Source']}, Order Destination: {order['Order_Destination']}, Distance (Kitchens): {order['Distance_Between_Kitchens']}, Ready Time Diff: {order['Ready_Time_Difference']}, Assigned Rule: {order['assigned_rule']}")
