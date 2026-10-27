import numpy as np


class MultiFeatureLinearModel:
    learning_rate: float
    bias: float
    weights: np.ndarray


    def __init__(self, num_features, learning_rate):
        self.weights = np.full(num_features, 0.0, dtype=float)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, x) -> np.ndarray:
        return x @ self.weights + self.bias

    def train_step(self, x, y):
        predictions = self.predict(x)
        errors = predictions - y

        avg_weight_gradient = (2 / x.shape[0]) * (x.T @ errors)
        avg_bias_gradient = np.mean(errors * 2)

        self.weights = self.weights - (self.learning_rate * avg_weight_gradient)
        self.bias = self.bias - (self.learning_rate * avg_bias_gradient)