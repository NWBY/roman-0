import numpy as np

from utils import (
    calculate_bias_gradient,
    calculate_weight_gradient,
    mean_squared_error,
    print_divider,
    squared_loss,
)


class Model:
    name: str
    weight: float
    bias: float
    learning_rate: float
    dataset: list[tuple[float, float]]
    validation: list[tuple[float, float]]
    test: list[tuple[float, float]]
    before_mse_str: str
    validation_mse: float

    def __init__(
        self,
        name: str,
        weight: float,
        bias: float,
        learning_rate: float,
        dataset: list[tuple[float, float]],
        validation: list[tuple[float, float]],
        test: list[tuple[float, float]],
    ):
        self.name = name
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.validation = validation
        self.test = test

    def train_model(self):
        self.before_mse_str = f"{self.name} Dataset MSE before: {mean_squared_error(self.dataset, self.weight, self.bias)}"
        print(f"Training {self.name}...")
        for i in range(1000):
            self.train_step_dataset()

        return self.weight, self.bias

    def train_model_numpy(self):
        self.before_mse_str = f"{self.name} Dataset MSE before: {mean_squared_error(self.dataset, self.weight, self.bias)}"
        print(f"Training {self.name}...")
        for i in range(1000):
            self.train_step_numpy()

        return self.weight, self.bias

    def train_step_dataset(self):
        weight_gradients = []
        bias_gradients = []
        for x, y in self.dataset:
            weight_gradients.append(
                calculate_weight_gradient(x, y, self.weight, self.bias)
            )
            bias_gradients.append(calculate_bias_gradient(x, y, self.weight, self.bias))

        avg_weight_gradient = sum(weight_gradients) / len(self.dataset)
        avg_bias_gradient = sum(bias_gradients) / len(self.dataset)

        self.weight = self.weight - self.learning_rate * avg_weight_gradient
        self.bias = self.bias - self.learning_rate * avg_bias_gradient

        return self.weight, self.bias

    def train_step_numpy(self):
        x = np.array([x[0] for x in self.dataset])
        y = np.array([y[1] for y in self.dataset])
        predictions = self.weight * x + self.bias
        errors = predictions - y
        squared_errors = errors**2

        avg_bias_gradient = np.mean(errors * 2)
        avg_weight_gradient = np.mean(errors * 2 * x)

        self.weight = self.weight - (self.learning_rate * avg_weight_gradient)
        self.bias = self.bias - (self.learning_rate * avg_bias_gradient)

        return self.weight, self.bias

    def train_step_multi_feature(self, X, y, w, b, learning_rate):
        predictions = X @ w + b
        errors = predictions - y

        avg_weight_gradient = (2 / X.shape[0]) * (X.T @ errors)
        avg_bias_gradient = np.mean(errors * 2)

        w = w - (learning_rate * avg_weight_gradient)
        b = b - (learning_rate * avg_bias_gradient)

        return w, b

        
    def validate_model(self):
        self.validation_mse = mean_squared_error(
            self.validation, self.weight, self.bias
        )

    def print_summary(self):
        print_divider()
        print("SUMMARY")
        print_divider()
        print(self.before_mse_str)
        print(f"Name: {self.name}")
        print(f"Final weight: {self.weight}")
        print(f"Final bias: {self.bias}")
        print(
            "Dataset MSE after:",
            mean_squared_error(self.dataset, self.weight, self.bias),
        )
        print(f"Validation MSE: {self.validation_mse}")
        # print("Test MSE:", mean_squared_error(test, w, b))

    def get_training_mse(self):
        return mean_squared_error(self.dataset, self.weight, self.bias)
