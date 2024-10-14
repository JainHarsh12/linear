import pytest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@pytest.fixture
def dataset():
    """Fixture for loading and splitting the dataset."""
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def model():
    """Fixture for initializing the Linear Regression model."""
    return LinearRegression()

def test_dataset_shape(dataset):
    """Test that the dataset has the correct dimensions after splitting."""
    X_train, X_test, y_train, y_test = dataset
    assert X_train.shape[0] > 0, "Training set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions do not match between train and test sets"
    assert y_train.shape[0] == X_train.shape[0], "Mismatch between X_train and y_train"
    assert y_test.shape[0] == X_test.shape[0], "Mismatch between X_test and y_test"

def test_model_training(model, dataset):
    """Test that the model can be trained on the dataset."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    assert model.coef_.shape[0] == X_train.shape[1], "The number of model coefficients does not match the number of features"
    assert model.intercept_ is not None, "Model intercept was not set"

def test_predictions_shape(model, dataset):
    """Test that the model's predictions have the correct shape."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predictions shape does not match the target shape"

def test_model_performance(model, dataset):
    """Test that the model has a reasonable R-squared value."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.4, f"R-squared is too low: {r2}"

def test_mean_squared_error(model, dataset):
    """Test that the model's Mean Squared Error is within an acceptable range."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 4000, f"Mean Squared Error is too high: {mse}"
