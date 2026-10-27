import numpy as np

def print_summary(w, b, training, validation, test, before_mse_str):
    print_divider()
    print("SUMMARY")
    print_divider()
    print(before_mse_str)
    print(f"Final weight: {w}, Final bias: {b}")
    print("Dataset MSE after:", mean_squared_error(training, w, b))
    print("Validation MSE:", mean_squared_error(validation, w, b))
    # print("Test MSE:", mean_squared_error(test, w, b))


def print_divider():
    print("--------------------------------")


def squared_loss(x, y, w, b):
    return (w * x + b - y) ** 2

def multi_errors(x, y, w, b):
    predictions = x @ w + b
    return predictions - y

def mean_squared_error(examples, w, b):
    # Calculate the squared loss for each example
    # Return the average squared loss
    return sum(squared_loss(x, y, w, b) for x, y in examples) / len(examples)

def mean_squared_error_multi(x, y, w, b):
    # Calculate the squared loss for each example
    # Return the average squared loss
    return np.mean(multi_errors(x, y, w, b) ** 2)


def calculate_weight_gradient(x, y, w, b):
    prediction = w * x + b
    error = prediction - y
    slope_loss = error * 2
    return slope_loss * x


def calculate_bias_gradient(x, y, w, b):
    prediction = w * x + b
    error = prediction - y
    slope_loss = error * 2
    return slope_loss * 1
