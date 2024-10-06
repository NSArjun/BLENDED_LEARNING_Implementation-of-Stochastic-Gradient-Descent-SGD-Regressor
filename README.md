# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Import the required libraries.

2. **Load Dataset**:  
   Load the dataset into your environment.

3. **Data Preprocessing**:  
   Handle missing values and encode categorical variables.

4. **Define Features and Target**:  
   Separate the data into features (X) and the target variable (y).

5. **Split Data**:  
   Divide the data into training and testing sets.

6. **Initialize SGD Regressor**:  
   Create an SGD Regressor model.

7. **Train the Model**:  
   Fit the model on the training dataset.

8. **Evaluate Performance**:  
   Assess the model's performance using evaluation metrics.

9. **Make Predictions & Visualize**:  
   Make predictions and visualize the results.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Aaron H
RegisterNumber: 212223040001
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the SGD Regressor model
model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Visualizing actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
```

## Output:
<img width="780" alt="Screenshot 2024-10-06 at 8 48 56 PM" src="https://github.com/user-attachments/assets/02eb9acd-52d8-4e8e-bc89-bb080032cfb9">




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
