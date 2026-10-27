import numpy as np
from multi_model import MultiFeatureLinearModel
from utils import mean_squared_error_multi


if __name__ == "__main__":
    validation_mses = []
    test_mses = []
    for i in range(10):
        rng = np.random.default_rng(i)
        x = rng.uniform(0, 3, size=(100, 2))
        p = rng.permutation(len(x))
        y = 2 * x[:, 0] + 3 * x[:, 1] + 1
        noise = rng.normal(0, 0.3, size=100)
        y = y + noise

        x_shuffled = x[p]
        y_shuffled = y[p]

        training_x = x_shuffled[:70]
        training_y = y_shuffled[:70]
        val_x = x_shuffled[70:85]
        val_y = y_shuffled[70:85]
        test_x = x_shuffled[85:]
        test_y = y_shuffled[85:]

        model = MultiFeatureLinearModel(
            num_features=2,
            learning_rate=0.05
        )

        print(f"Multi MSE before: {mean_squared_error_multi(training_x, training_y, model.weights, model.bias)}")
        for _ in range(1000):
            model.train_step(training_x, training_y)
        print(f"Multi MSE after: {mean_squared_error_multi(training_x, training_y, model.weights, model.bias)}")

        validation_mse = mean_squared_error_multi(val_x, val_y, model.weights, model.bias)
        print(f"Multi validation MSE: {validation_mse}")
        validation_mses.append(validation_mse)

        test_mse = mean_squared_error_multi(test_x, test_y, model.weights, model.bias)
        print(f"Multi test MSE: {test_mse}")
        test_mses.append(test_mse)

    print(f"Average validation MSE: {np.mean(validation_mses)}")
    print(f"Average test MSE: {np.mean(test_mses)}")
    print(f"test RMSE: {np.sqrt(np.mean(test_mses))}")