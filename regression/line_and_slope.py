import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate sample data
np.random.seed(0)
X = np.random.rand(100)  # 100 random points between 0 and 1
y = 3 * X + 2 + np.random.randn(100) * 0.1  # y = 3x + 2 + noise

# Step 2: Implement Linear Regression
def linear_regression(X, y):
    # Calculate mean of X and y
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate the slope (m)
    numerator = sum((X - X_mean) * (y - y_mean))
    denominator = sum((X - X_mean)**2)
    m = numerator / denominator
    
    # Calculate the intercept (b)
    b = y_mean - m * X_mean
    
    return m, b

# Train the model
m, b = linear_regression(X, y)

# Step 3: Predict and visualize
y_pred = m * X + b

plt.scatter(X, y, color='blue', label='Data Points')
# plt.plot(X, y_pred, color='red', label=f'Best Fit Line: y = {m:.2f}x + {b:.2f}')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression from Scratch')
plt.show()
