import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])
Y = np.array([0, 1, 1, 1])  # OR gate outputs
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
epochs = 10
def activation_fn(x):
    return 1 if x >= 0 else 0
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_predicted = activation_fn(linear_output)
        error = Y[i] - y_predicted
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        print(f"Input: {X[i]}, Predicted: {y_predicted}, Error: {error}, Updated Weights: {weights}, Updated Bias: {bias}")
print("\nFinal Weights:", weights)
print("Final Bias:", bias)
for i in range(len(X)):
    if Y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', marker='o')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', marker='x')
x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
y_values = -(weights[0] * np.array(x_values) + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Single Layer Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show() 
