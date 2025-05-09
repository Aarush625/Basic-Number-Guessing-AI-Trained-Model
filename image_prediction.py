import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import requests
from io import BytesIO

# Load the trained model
model = np.load('trained_model.npz')
W1, b1 = model['W1'], model['b1']
W2, b2 = model['W2'], model['b2']
W3, b3 = model['W3'], model['b3']

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Predict function
def predict_digit(img_array):
    img_array = img_array / 16.0  # normalize
    Z1 = np.dot(img_array, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return np.argmax(A3, axis=1)[0]

# Fetch image from URL and preprocess
def fetch_and_predict(image_path_or_url):
    # Load model
    model = np.load('trained_model.npz')
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    W3, b3 = model['W3'], model['b3']

    # Check if input is a URL or local path
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    else:
        img = Image.open(image_path_or_url).convert('L')  # Local image

    # Resize and invert image
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)

    # Convert to numpy array and normalize like training data
    img_array = np.array(img).reshape(1, -1)
    img_array = img_array / 16.0  # Normalize to match training

    # Forward pass
    A1 = sigmoid(np.dot(img_array, W1) + b1)
    A2 = sigmoid(np.dot(A1, W2) + b2)
    A3 = softmax(np.dot(A2, W3) + b3)

    # Prediction
    prediction = np.argmax(A3)
    print(f"Predicted Digit: {prediction}")

    plt.imshow(img_array.reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {prediction}")
    plt.show()


# Example usage
image_url = "./seven.png"  # Replace with your image URL
fetch_and_predict(image_url)
