# ROMAN-0 — From-scratch linear regression

My first machine-learning model: learning linear regression, gradients, training and evaluation by implementing them in Python and NumPy.

ROMAN-0 is the first linear regression model I've built and trained.

It started with a single `Model` class. The latest version now uses a separate `MultiFeatureLinearModel` class with `predict()` and `train_step()` methods. The model owns its weights, bias and learning rate, while the training script handles the dataset, training loop and evaluation. Gradient descent is used to adjust the weights and bias and reduce loss.

The project started in pure Python using loops and manual calculations, then moved on to NumPy and multiple input features and weights.

The two training files are `train.py` and `multi.py`. `train.py` contains a lot of the earlier manual Python work and `multi.py` is the latest experiment, training on two input features with added noise. The new model itself is in `multi_model.py`.

This is a learning project, not a production forecasting system. The model's gradients and training updates are implemented directly rather than delegated to a machine-learning training framework.

## Running it

You'll need Python 3.11+ and [uv](https://docs.astral.sh/uv/) installed. From the repository root:

```bash
uv run multi.py
```

This runs the latest experiment and uses NumPy for the calculations. To explore the earlier single-feature implementation (including hand-written Python gradient updates), run `uv run train.py`.

To run the checks for predictions and a single training step:

```bash
uv run multi_model_test.py
```

The tests use NumPy assertions to check the predictions, weights and bias. They also check that the bias actually changes when its gradient isn't zero. The test script exits without an error when everything passes.

## The dataset

I'm generating the data myself for now. Each observation has two input features with values randomly generated between 0 and 3. The target is calculated using:

```python
y = 2 * x1 + 3 * x2 + 1 + noise
```

The noise is normally distributed with a mean of 0 and standard deviation of 0.3. Because I know the underlying relationship, I can see whether the model learns weights close to 2 and 3 and a bias close to 1.

Each dataset contains 100 observations. I shuffle the inputs and targets together, then split them into 70 for training, 15 for validation and 15 for testing.

The experiment runs 10 times with different random seeds (0–9). Each run starts with weights and bias set to zero and trains for 1,000 steps using a learning rate of 0.05. Only the training observations are used to update the model. The validation and test observations are held out, with the test set used for the final evaluation.

## Results

Across the 10 runs:

| Metric | Result |
| --- | ---: |
| Average validation MSE | 0.1048 |
| Average test MSE | 0.0940 |
| Overall test RMSE | 0.3066 |

The test RMSE is close to the 0.3 standard deviation of the noise I added, which suggests the model has learned the underlying relationship reasonably well.

These results are only from synthetic data generated using the same formula and noise distribution. I haven't tested the model on real-world data or where the underlying relationship changes, so I wouldn't claim it would perform the same outside this experiment.

## License

MIT — see [LICENSE](LICENSE).
