import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
class LinearRegression_GradientDescent:
    def __init__(self, lr=0.02, epochs=1000):
        self.lr = lr  # Learning rate (how big each step is)
        self.epochs = epochs  # Number of times to go through the data
        self.w = None  # Weights (to be learned)
        self.b = None  # Bias (to be learned)
        self.mean = None  # Mean for scaling
        self.std = None  # Standard deviation for scaling
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get number of rows and columns
        self.w = np.zeros(n_features)  # Start weights at zero
        self.b = 0  # Start bias at zero
        self.mean = np.mean(X, axis=0)  # Find mean of each column
        self.std = np.std(X, axis=0)  # Find std of each column
        X_scaled = (X - self.mean) / self.std  # Scale features
        for i in range(self.epochs):  # Repeat for set number of times
            y_pred = np.dot(X_scaled, self.w) + self.b  # Predict output
            dw = (2 / n_samples) * np.dot(X_scaled.T, (y_pred - y))  # Find gradient for weights
            db = (2 / n_samples) * np.sum(y_pred - y)  # Find gradient for bias
            self.w -= self.lr * dw  # Update weights
            self.b -= self.lr * db  # Update bias
        return self  # Return the trained model
    def predict(self, X):
        X_scaled = (X - self.mean) / self.std  # Scale input features
        return np.dot(X_scaled, self.w) + self.b  # Return predictions
    def score(self, X, y):
        y_pred = self.predict(X)  # Get predictions
        return r2_score(y, y_pred)  # Return R2 score
# Linear Regression using Ordinary Least Squares (OLS)
class OLSLinearRegression:
    def __init__(self):
        self.w = None  # Weights
        self.b = None  # Bias
    def fit(self, X, y):
        X = np.array(X)  # Make sure X is a numpy array
        y = np.array(y)  # Make sure y is a numpy array
        X_mean = X.mean(axis=0)  # Mean of features
        y_mean = y.mean()  # Mean of target
        A = X.T @ X  # Matrix multiplication for OLS
        b_vec = X.T @ (y - y_mean)  # Another part of OLS formula
        self.w = np.linalg.solve(A, b_vec)  # Solve for weights
        self.b = y_mean - X_mean @ self.w  # Solve for bias
    def predict(self, X):
        X = np.array(X)  # Make sure X is a numpy array
        return X @ self.w + self.b  # Return predictions
# Load the dataset from CSV file
dataset = pd.read_csv("data.csv")
# Remove columns that are not needed
dataset.drop(["dteday", "instant","casual","registered"], axis=1, inplace=True)
# Get all columns except last as features
X = dataset.iloc[:, :-1].values
# Get last column as target
y = dataset.iloc[:, -1].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Create the gradient descent model
model = LinearRegression_GradientDescent(lr=0.2, epochs=1000)
# Train the model
model.fit(X_train, y_train)
print("GRADIENT DESCENT METHOD")
print("------------------------------")
print("Trained Weights:", model.w)  # Show learned weights
print("Trained Bias:", model.b)  # Show learned bias
print("------------------------------")
# Make predictions on test data
predictions = model.predict(X_test)
print("Predictions:", predictions[:10])  # Show first 10 predictions
print("Actual values:", y_test[:10])  # Show first 10 actual values
print("R2 SCORE:", r2_score(y_test, predictions))  # Show R2 score
print("MSE:", mean_squared_error(y_test, predictions))  # Show mean squared error
# Create the OLS model
model1=OLSLinearRegression()
# Train the OLS model
model1.fit(X_train, y_train)
print("-------------------------------")
print("OLS METHOD")
print("------------------------------")
print("Trained Weights:", model1.w)  # Show OLS weights
print("Trained Bias:", model1.b)  # Show OLS bias
print("------------------------------")
# Make predictions with OLS
predictions = model1.predict(X_test)
print("Predictions:", predictions[:10])  # Show first 10 predictions
print("Actual values:", y_test[:10])  # Show first 10 actual values
print("R2 SCORE:", r2_score(y_test, predictions))  # Show R2 score
print("MSE:", mean_squared_error(y_test, predictions))  # Show
