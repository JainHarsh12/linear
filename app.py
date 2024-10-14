# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the diabetes dataset
diabetes = load_diabetes()

# Splitting the dataset into features (X) and target (y)
X = diabetes.data  # Features
y = diabetes.target  # Target variable (diabetes progression)

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model on the training data
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2 Score):", r2)

# Displaying the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
