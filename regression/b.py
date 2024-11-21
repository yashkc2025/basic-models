import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 random points between 0 and 1
y = 4 * (X**2) + 3 * X + 2 + np.random.randn(100, 1) * 0.2  # Quadratic equation with noise

# Add bias term for polynomial regression (X^2 term)
X_poly = np.hstack([X**2, X, np.ones_like(X)])

# Step 2: Define helper functions
def predict(X, weights):
    """Calculate predictions: y = X @ weights."""
    return np.dot(X, weights)

def mean_squared_error(y, y_pred):
    """Compute Mean Squared Error (MSE)."""
    return np.mean((y - y_pred)**2)

def gradient_descent(X, y, weights, lr=0.01):
    """Perform one step of gradient descent."""
    n = len(y)
    y_pred = predict(X, weights)
    gradients = -2/n * np.dot(X.T, (y - y_pred))  # Partial derivatives of MSE
    weights -= lr * gradients  # Update weights
    return weights

# Step 3: Train the model
weights = np.random.randn(3, 1)  # Initialize random weights for [a, b, c]
learning_rate = 0.1
epochs = 1000
errors = []

for epoch in range(epochs):
    weights = gradient_descent(X_poly, y, weights, lr=learning_rate)
    y_pred = predict(X_poly, weights)
    error = mean_squared_error(y, y_pred)
    errors.append(error)

# Step 4: Evaluate the model
print(f"Trained weights: {weights.ravel()}")
print(f"Final MSE: {errors[-1]:.4f}")

# Plot training error over epochs
plt.figure(figsize=(10, 5))
plt.plot(errors)
plt.title("Training Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Step 5: Visualize the model's predictions
plt.scatter(X, y, label="Data")
plt.plot(np.sort(X, axis=0), predict(np.sort(X_poly, axis=0), weights), color='red', label="Model Prediction")
plt.title("Polynomial Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
