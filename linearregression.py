from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -----------------------------
# Dataset
# -----------------------------
X = np.array([[1], [2], [3], [4], [5], [6], [7]])  # hours of deep work
y = np.array([2, 4, 5, 4, 5, 7, 8])                # tasks completed

# -----------------------------
# Model Training
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# Predictions on training data
y_pred = model.predict(X)

# -----------------------------
# Evaluation
# -----------------------------
mse = mean_squared_error(y, y_pred)

print("Mean Squared Error:", mse)
print("Actual values:", y)
print("Predicted values:", y_pred)

# -----------------------------
# Custom Prediction
# -----------------------------
hours = int(input("\nEnter hours of deep work: "))
predicted_value = model.predict([[hours]])

print(f"Predicted tasks completed for {hours} hours:", predicted_value[0])

# -----------------------------
# Model Parameters
# -----------------------------
print("\nModel Slope (Coefficient):", model.coef_)
print("Model Intercept:", model.intercept_)

# -----------------------------
# Visualization
# -----------------------------
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color="teal", label="Best Fit Line")

plt.xlabel("Hours of Deep Work")
plt.ylabel("Tasks Completed")
plt.title("Linear Regression: Deep Work vs Productivity")
plt.legend()
plt.show()
