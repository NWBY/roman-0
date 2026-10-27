import random
import numpy as np
from dataset import generate_examples, shuffle_examples, generate_rnd
from utils import mean_squared_error, mean_squared_error_multi
from model import Model


def train_step(x, y, w, b, learning_rate):
    prediction = w * x + b
    error = prediction - y
    slope_loss = error * 2
    weight_gradient = slope_loss * x
    bias_gradient = slope_loss * 1

    new_w = w - learning_rate * weight_gradient
    new_b = b - learning_rate * bias_gradient

    return new_w, new_b


if __name__ == "__main__":
    _, _, w, b, learning_rate = 2.0, 5.0, 2.0, 0.0, 0.05
    ten_models = []
    thirty_models = []
    seventy_models = []

    rnd = generate_rnd(42)

    dataset = generate_examples(rnd, 100, 0, 3)
    shuffle_examples(rnd, dataset)

    validation = dataset[70:85]
    test = dataset[85:]

    numpy_model = Model(
        name=f"Model np",
        weight=2.0,
        bias=0.0,
        learning_rate=0.05,
        dataset=dataset[:70],
        validation=validation,
        test=test,
    ) 
    std_model = Model(
        name=f"Model std",
        weight=2.0,
        bias=0.0,
        learning_rate=0.05,
        dataset=dataset[:70],
        validation=validation,
        test=test,
    ) 

    numpy_model.train_model_numpy()
    std_model.train_model()

    numpy_model.validate_model()
    std_model.validate_model()

    np_model_w = numpy_model.weight
    np_model_b = numpy_model.bias

    std_model_w = std_model.weight
    std_model_b = std_model.bias

    print(f"np weight: {np_model_w}")
    print(f"std weight: {std_model_w}")
    print(f"np bias: {np_model_b}")
    print(f"std bias: {std_model_b}")
    print(f"np training mse: {numpy_model.get_training_mse()}")
    print(f"std training mse: {std_model.get_training_mse()}")

    print(f"is weight close: {np.isclose(np_model_w, std_model_w)}")
    print(f"is bias close: {np.isclose(np_model_b, std_model_b)}")
    print(f"is training mse close: {np.isclose(numpy_model.get_training_mse(), std_model.get_training_mse())}")

    X = np.array([[2, 3], [1, 4], [0, 2]])
    y = np.array([22, 28, 10])

    w = np.array([4.0, 5.0])
    b = 1.0

    learning_rate = 0.05

    multi_model = Model(
        name=f"Multi model",
        weight=w,
        bias=b,
        learning_rate=0.05,
        dataset=X,
        validation=validation,
        test=test, 
    )

    print(f"Multi MSE before: {mean_squared_error_multi(X, y, w, b)}")
    for i in range(1000):
        w, b = multi_model.train_step_multi_feature(
            X, y, w, b, learning_rate
        )
    print(f"Multi MSE after: {mean_squared_error_multi(X, y, w, b)}")

    print(f"weights: {w}")
    print(f"bias: {b}")

    """
    for i in range(10):
        print(f"Training model {i}...")
        rnd = generate_rnd(i)

        dataset = generate_examples(rnd, 100, 0, 3)
        shuffle_examples(rnd, dataset)

        validation = dataset[70:85]
        test = dataset[85:]

        model_a = Model(
            name=f"Model {i} - Training: 10, Validation: 15, Test: 15",
            weight=2.0,
            bias=0.0,
            learning_rate=0.05,
            dataset=dataset[:10],
            validation=validation,
            test=test,
        )

        model_b = Model(
            name=f"Model {i} - Training: 30, Validation: 15, Test: 15",
            weight=2.0,
            bias=0.0,
            learning_rate=0.05,
            dataset=dataset[:30],
            validation=validation,
            test=test,
        )

        model_c = Model(
            name=f"Model {i} - Training: 70, Validation: 15, Test: 15",
            weight=2.0,
            bias=0.0,
            learning_rate=0.05,
            dataset=dataset[:70],
            validation=validation,
            test=test,
        )

        model_a.train_model()
        model_b.train_model()
        model_c.train_model()

        model_a.validate_model()
        model_b.validate_model()
        model_c.validate_model()

        ten_models.append(model_a)
        thirty_models.append(model_b)
        seventy_models.append(model_c)

    for model in ten_models:
        model.print_summary()

    for model in thirty_models:
        model.print_summary()

    for model in seventy_models:
        model.print_summary()

    ten_average_validation_mse = sum([m.validation_mse for m in ten_models]) / len(
        ten_models
    )
    thirty_average_validation_mse = sum(
        [m.validation_mse for m in thirty_models]
    ) / len(thirty_models)
    seventy_average_validation_mse = sum(
        [m.validation_mse for m in seventy_models]
    ) / len(seventy_models)

    print(f"Average validation MSE for 10: {ten_average_validation_mse}")
    print(f"Average validation MSE for 30: {thirty_average_validation_mse}")
    print(f"Average validation MSE for 70: {seventy_average_validation_mse}")
    """

    # print("Testing Model...")
    # print_divider()
    # for x, y in test:
    #     test_prediction = w * x + b
    #     test_error = test_prediction - y
    #     test_squared_loss = squared_loss(x, y, w, b)

    #     print(
    #         f"x: {x}, y: {y}, test prediction: {test_prediction}, test error: {test_error}, test squared loss: {test_squared_loss}"
    #     )
    #     print("")
    # print("Test MSE:", mean_squared_error(test, w, b))
    # print_divider()
