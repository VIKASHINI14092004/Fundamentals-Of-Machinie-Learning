import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate coefficients
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

# Prediction function
def predict(x_val):
    return slope * x_val + intercept

# Predict y values
y_pred = predict(x)

# Print results
print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")
print("Regression Line Equation: y = {:.2f}x + {:.2f}".format(slope, intercept))

# Plotting
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression using Least Squares Method')
plt.legend()
plt.grid(True)
plt.show()
