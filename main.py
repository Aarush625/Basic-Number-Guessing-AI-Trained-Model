import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target.reshape(-1, 1)

# Normalize the input data
X = X / 16.0

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Seed
np.random.seed(1)

# Network structure
input_size = 64
hidden_size = 64
hidden_size2 = 32
output_size = 10

# Hyperparameters
epochs = 10000
lr = 0.0001
model_file = 'trained_model.npz'

# Track losses and accuracy
train_losses = []
test_accuracies = []

# Try to load saved model
try:
    model = np.load(model_file)
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']
    print("Loaded saved model weights.")
except FileNotFoundError:
    print("Training a new model.")

    # Weight Initialization
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, hidden_size2) * 0.1
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, output_size) * 0.1
    b3 = np.zeros((1, output_size))

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Forward pass (training data)
        Z1 = np.dot(X_train, W1) + b1
        A1 = sigmoid(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        # Compute training loss
        error = y_train - A3
        loss = -np.mean(np.sum(y_train * np.log(A3 + 1e-10), axis=1))
        train_losses.append(loss)

        # Backpropagation
        dA3 = error
        dW3 = np.dot(A2.T, dA3)
        db3 = np.sum(dA3, axis=0, keepdims=True)

        dA2 = np.dot(dA3, W3.T) * sigmoid_derivative(A2)
        dW2 = np.dot(A1.T, dA2)
        db2 = np.sum(dA2, axis=0, keepdims=True)

        dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
        dW1 = np.dot(X_train.T, dA1)
        db1 = np.sum(dA1, axis=0, keepdims=True)

        # Update weights
        W3 += lr * dW3
        b3 += lr * db3
        W2 += lr * dW2
        b2 += lr * db2
        W1 += lr * dW1
        b1 += lr * db1

        # Log loss
        train_losses.append(loss)

        # Evaluate test accuracy every 100 epochs
        if epoch % 100 == 0:
            Z1_test = np.dot(X_test, W1) + b1
            A1_test = sigmoid(Z1_test)
            Z2_test = np.dot(A1_test, W2) + b2
            A2_test = sigmoid(Z2_test)
            Z3_test = np.dot(A2_test, W3) + b3
            A3_test = softmax(Z3_test)

            predictions = np.argmax(A3_test, axis=1)
            actual = np.argmax(y_test, axis=1)
            accuracy = np.mean(predictions == actual)
            test_accuracies.append(accuracy)

            print(f"Epoch {epoch} Loss: {loss:.4f} | Test Accuracy: {accuracy * 100:.2f}%")

            # Save the model
            np.savez(model_file, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

    # Ensure test_accuracies is populated before plotting
    if len(test_accuracies) == 0:
        print("Warning: Test accuracies list is empty.")
    else:
        # Plotting the results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, len(test_accuracies)*100, 100), test_accuracies, label='Test Accuracy', color='green')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# --- Testing ---
Z1 = np.dot(X_test, W1) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
Z3 = np.dot(A2, W3) + b3
A3 = softmax(Z3)

predictions = np.argmax(A3, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == actual)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Display a prediction
def show_prediction(index):
    plt.imshow(X_test[index].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {predictions[index]}")
    plt.axis('off')
    plt.show()

# Example display
show_prediction(0)