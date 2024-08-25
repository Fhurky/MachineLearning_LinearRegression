# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:38:10 2024

@author: furko
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset and select the relevant columns
data = pd.read_csv("Student_Performance.csv")
data = data[['Previous Scores', 'Performance Index']]

# Split the dataset into training and testing sets (80% training, 20% testing)
train, test = train_test_split(data, test_size=0.33, random_state=42)

# Separate the target variable (Performance Index) from the features
train_results = train.iloc[:,-1]
test_results = test.iloc[:,-1]

# Drop the target variable from the features
train = train.drop(['Performance Index'], axis=1)
test = test.drop(['Performance Index'], axis=1)

# Create and train the model
lm = LinearRegression()
lm.fit(train, train_results)

# Make predictions on the test data
prediction = lm.predict(test)

# Print the R² score
print("R2-score: %.2f" % r2_score(test_results, prediction))

# Plot the predictions against the actual results
plt.figure(figsize=(10, 10))
plt.scatter(test_results, prediction, color='blue')
plt.title('Student Score vs. Previous Scores')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.grid(True)
plt.show()
