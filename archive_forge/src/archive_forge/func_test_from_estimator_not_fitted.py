import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
def test_from_estimator_not_fitted(pyplot):
    """Check that we raise a `NotFittedError` when the passed regressor is not
    fit."""
    regressor = Ridge()
    with pytest.raises(NotFittedError, match='is not fitted yet.'):
        PredictionErrorDisplay.from_estimator(regressor, X, y)