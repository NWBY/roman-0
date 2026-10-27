import numpy as np
from multi_model import MultiFeatureLinearModel

if __name__ == "__main__":
    x = np.array([[2, 3], [1, 4], [0, 2]])
    y = np.array([22, 28, 10])

    model = MultiFeatureLinearModel(
        num_features=2,
        learning_rate=0.05
    )
    model.weights = np.array([4.0, 5.0])
    model.bias = 1.0

    predictions = model.predict(x)

    np.testing.assert_allclose(predictions, [24, 25, 11])
    assert predictions.shape == (3,)

    model.train_step(x, y)

    np.testing.assert_allclose(model.weights, [3.96666667, 5.13333333])
    np.testing.assert_allclose(model.bias, 1.0)

    y_a = np.array([27, 21, 17])

    model_1 = MultiFeatureLinearModel(
        num_features=2,
        learning_rate=0.05
    )
    model_1.weights = np.array([4.0, 5.0])
    model_1.bias = 1.0

    predictions_1 = model_1.predict(x)

    np.testing.assert_allclose(predictions_1, [24, 25, 11])
    assert predictions_1.shape == (3,)

    model_1.train_step(x, y_a)

    np.testing.assert_allclose(model_1.weights, [4.066666666666666, 5.166666666666667])
    np.testing.assert_allclose(model_1.bias, 1.1666666666666667)