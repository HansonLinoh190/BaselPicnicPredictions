import pytest
import joblib
import numpy as np

@pytest.fixture
def model():
    # Load the trained model
    return joblib.load("ML/Session 10/picnic_model.pkl")

def test_model_loaded(model):
    assert model is not None

def test_prediction_shape(model):
    # Data from 2000-01-01
    sample_data = np.array([[1, 8, 0.89, 10.286, 0.2, 0.03, 0.0, 2.9, 1.6, 3.9]])
    prediction = model.predict(sample_data)
    assert prediction.shape == (1,)

def test_known_negative_prediction(model):
    # Data from 2000-01-01: Cloudy, cold, rainy. Expected Res: FALSE (0)
    sample_data = np.array([[1, 8, 0.89, 10.286, 0.2, 0.03, 0.0, 2.9, 1.6, 3.9]])
    prediction = model.predict(sample_data)
    assert prediction[0] == 0

def test_known_positive_prediction(model):
    # Data from 2000-03-09: Clearer, warm (14.1C). Expected Res: TRUE (1)
    sample_data = np.array([[3, 7, 0.53, 10.278, 1.15, 0.0, 3.9, 14.1, 10.1, 19.3]])
    prediction = model.predict(sample_data)
    assert prediction[0] == 1

@pytest.mark.parametrize(
    "features, expected_class",
    [
        ([1, 8, 0.89, 10.286, 0.2, 0.03, 0.0, 2.9, 1.6, 3.9], 0),  #False
        ([3, 7, 0.53, 10.278, 1.15, 0.0, 3.9, 14.1, 10.1, 19.3], 1), #True
    ]
)
def test_mult_preds(model, features, expected_class):
    sample_data = np.array([features])
    prediction = model.predict(sample_data)
    assert prediction[0] == expected_class

@pytest.mark.parametrize(
    "features",
    [
        ([1, 8, 0.89, 10.286, 0.2, 0.03, 0.0, 2.9, 1.6, 3.9]),
        ([3, 7, 0.53, 10.278, 1.15, 0.0, 3.9, 14.1, 10.1, 19.3]),
    ]
)
def test_prob_mult_preds(model, features):
    sample_data = np.array([features])
    probabilities = model.predict_proba(sample_data)
    total_prob = probabilities[0].sum()
    assert total_prob == pytest.approx(1.0, abs=1e-6)