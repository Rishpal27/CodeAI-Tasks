#import necessary libraries
import numpy as np
import pandas as pd  

# Define the Regression Model
class LinearRegression:
    def __init__(self, lr=0.2, epochs=1000):
        self.lr = lr  # Learning rate for gradient descent
        self.epochs = epochs  # Number of training epochs/iterations
        self.w = None  # Weights
        self.b = None  # Bias 
        self.mean = None  # Mean for feature scaling
        self.std = None  # Standard deviation for feature scaling

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get number of data points and features
        self.w = np.zeros(n_features)  # Initialize weights to zeros for now
        self.b = 0  # Initialize bias to zero for now
        self.mean = np.mean(X, axis=0)  # Compute mean for each feature
        self.std = np.std(X, axis=0)  # Compute standard deviation for each feature
        X_scaled = (X - self.mean) / self.std  # Scale features

        for i in range(self.epochs):
            y_pred = np.dot(X_scaled, self.w) + self.b  # Predict output
            dw = (2 / n_samples) * np.dot(X_scaled.T, (y_pred - y))  # Compute gradient for weights
            db = (2 / n_samples) * np.sum(y_pred - y)  # Compute gradient for bias
            self.w -= self.lr * dw  # Updating weights
            self.b -= self.lr * db  # Updating bias
            mse = np.mean((y - y_pred) ** 2)  # Calculate mean squared error
            print(f"Epoch {i}, MSE: {mse}")  # Print epoch/iteration and MSE

    def predict(self, X):
        X_scaled = (X - self.mean) / self.std  # Scale input datapoints
        return np.dot(X_scaled, self.w) + self.b  # Return predictions with learned weights and bias

    def R2score(self, X, y):
        y_pred = self.predict(X)  # Get predictions
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)  # Calculate R2 score
        return r2  # Return R2 score

dataset = pd.read_csv("data.csv")  # Load given dataset from CSV file
#dropping unnecessary columns required for training
dataset.drop("dteday", axis=1, inplace=True)
dataset.drop("instant", axis=1, inplace=True)
#retrieving features and target variable  
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  
model = LinearRegression(lr=0.2, epochs=1000)  # Create Regression model object
model.fit(X, y)  # Training the model with the dataset

print("------------------------------")  
print("Trained Weights:", model.w)  # Print trained weights
print("Trained Bias:", model.b)  # Print trained bias
print("------------------------------")

predictions = model.predict(X[:5])  # Predict first 5 samples
print("Predictions:", predictions)  # Print predictions
print("Actual values:", y[:5])  # Print actual values

accuracy_score = model.R2score(X, y)  # Calculate R2 score
print("------------------------------")
print("R2 SCORE:", accuracy_score)  # Print R2 accuracy score
print("------------------------------")
