import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# [1] Data Pre-Processing

# Load the Iris dataset
iris = load_iris()
X = iris.data  
y = iris.target.reshape(-1, 1)

# One-hot encoding for target variable
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# [2] Activation function, loss function, NN architecture, and hyperparameters

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# MSE as loss/cost function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# NN Architecture
input_size = 4
hidden_size = 3
output_size = 3

# Hyperparameters
learning_rate = 0.05
iterations = (10 ** 5)

# [3] Initialize randomized weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# [4] NN Training Phase

# Training loop
losses = []

for i in range(iterations):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss calculation
    loss = mse(y_train, a2)
    losses.append(loss)

    # Backward pass
    error = y_train - a2
    d_a2 = error * sigmoid_derivative(a2)

    dW2 = np.dot(a1.T, d_a2)
    db2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, d_a1)
    db1 = np.sum(d_a1, axis=0, keepdims=True)

    # Update weights and biases
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
    W1 += learning_rate * dW1
    b1 += learning_rate * db1

# [5] Visualization of cost function over iterations (NN training)

# Plotting the cost function over iterations
plt.plot(range(iterations), losses)
plt.title("Iris Dataset -- Training Phase")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(False)
plt.show()

 

 
