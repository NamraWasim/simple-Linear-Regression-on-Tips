import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# tips dataset
data = {
    "total_bill": [16.99, 10.34, 21.01, 23.68, 24.59, 25.29, 8.77, 26.88, 15.04, 14.78],
    "tip": [1.01, 1.66, 3.5, 3.31, 3.61, 4.71, 2.0, 3.12, 1.96, 3.23],
    "sex": ["Female", "Male", "Male", "Male", "Female", "Male", "Male", "Male", "Male", "Male"],
    "smoker": ["No", "No", "No", "No", "No", "No", "No", "No", "No", "No"],
    "day": ["Sun", "Sun", "Sun", "Sun", "Sun", "Sun", "Sun", "Sun", "Sun", "Sun"],
    "time": ["Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner"],
    "size": [2, 3, 3, 2, 4, 2, 4, 2, 2, 2]
}
df = pd.DataFrame(data)
# Perform simple linear regression
X = df[['total_bill']]  # Feature (independent variable)
y = df['tip']  # Target (dependent variable)
model = LinearRegression()
model.fit(X, y)
# Visualize the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=df, color='blue', alpha=0.6)
plt.plot(X, model.predict(X), color='red', linewidth=2)  # Plotting the regression line
plt.title('Linear Regression of Tips on Total Bill')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.grid(True)
plt.show()
