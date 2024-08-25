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

data = pd.read_csv("Student_Performance.csv")
data = data.drop(['Extracurricular Activities'], axis = 1)

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_results = train.iloc[:,-1]
test_results = test.iloc[:,-1]

train = train.drop(['Performance Index'], axis = 1)
test = test.drop(['Performance Index'], axis = 1)

lm = LinearRegression()
lm.fit(train,train_results)

prediction = lm.predict(test)
print("R2-score: %.2f" % r2_score(test_results, prediction))


plt.figure(figsize=(10, 10))
plt.scatter(prediction, test_results, color='blue')
plt.title('Student Score vs. Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Performance Index')
plt.grid(True)
plt.show()


