import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = 0

    def sigmod(self, a):
        return 1 / (1 + np.exp(-a))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.num_iterations):
            prediction = self.sigmod(np.dot(X, self.weights) + self.bias)

            dw = (1 / n_samples) * np.dot(X.T, (prediction - y))
            db = (1 / n_samples) * np.sum(prediction - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predict = self.sigmod(np.dot(X, self.weights) + self.bias)
        return [1 if y > 0.5 else 0 for y in y_predict]