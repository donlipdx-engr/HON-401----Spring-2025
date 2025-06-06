# Build and train neural network on data from Higham (2019) (cf. Listing 6.1)
# Plot value of cost function as function of number of epochs

import numpy as np
import matplotlib.pyplot as plt

# [1] Use Sigmoid as activation function, MSE as cost/loss function

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Mean squared error (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivative of MSE
def mse_derivative(y_true, y_pred):
    return y_pred - y_true

# [2] Data Pre-Processing (cf. Listing 6.1 in Higham(2019))
x1 = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
x2 = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]
X = np.column_stack((x1, x2))  

arr1 = np.ones(5)
arr2 = np.zeros(5)
y_top = np.concatenate((arr1, arr2), axis=0)
y_bottom = np.concatenate((arr2, arr1), axis=0)
y = np.vstack((y_top, y_bottom)).T  

# [3] Intialize hyperparameters 

# NN architecture
input_size = 2
hidden1_size = 2
hidden2_size = 3
output_size = 2

# Hyperparameters: learning rate, epochs
learning_rate = 0.01
max_epochs = (10 ** 5)

# --- Weight & Bias Initialization ---
np.random.seed(42)
W1 = np.random.randn(input_size, hidden1_size)
b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(hidden1_size, hidden2_size)
b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(hidden2_size, output_size)
b3 = np.zeros((1, output_size))

# [4] Neural Network Training Phase

loss_history = []

epoch = 1
while (epoch <= max_epochs):
    total_loss = 0

    for i in range(X.shape[0]):
        x_sample = X[i:i+1]
        y_sample = y[i:i+1]

        # Forward pass
        Z1 = np.dot(x_sample, W1) + b1
        A1 = sigmoid(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoid(Z3)

        # Compute loss
        loss = mse(y_sample, A3)
        total_loss += loss

        # Backpropagation
        dz3 = mse_derivative(y_sample, A3) * sigmoid_derivative(Z3)
        dW3 = np.dot(A2.T, dz3)
        db3 = dz3

        dz2 = np.dot(dz3, W3.T) * sigmoid_derivative(Z2)
        dW2 = np.dot(A1.T, dz2)
        db2 = dz2

        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(Z1)
        dW1 = np.dot(x_sample.T, dz1)
        db1 = dz1

        # Update weights and biases
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    avg_loss = total_loss / X.shape[0]
    loss_history.append(avg_loss)

    if epoch % (10 ** 4) == 0:
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

    epoch += 1

# [5] Plot value of cost function as function of number of epochs
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Value of Cost Function')
plt.title('Neural Network Training Phase')
plt.grid(True)
plt.legend()
plt.show()




