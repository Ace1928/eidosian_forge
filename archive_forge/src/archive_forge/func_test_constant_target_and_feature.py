import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('y, y_mean', [(np.array([3.4] * 20), 3.4), (np.array([0] * 20), 0), (np.array(['a'] * 20, dtype=object), 0)], ids=['continuous', 'binary', 'binary-string'])
@pytest.mark.parametrize('smooth', ['auto', 4.0, 0.0])
def test_constant_target_and_feature(y, y_mean, smooth):
    """Check edge case where feature and target is constant."""
    X = np.array([[1] * 20]).T
    n_samples = X.shape[0]
    enc = TargetEncoder(cv=2, smooth=smooth, random_state=0)
    X_trans = enc.fit_transform(X, y)
    assert_allclose(X_trans, np.repeat([[y_mean]], n_samples, axis=0))
    assert enc.encodings_[0][0] == pytest.approx(y_mean)
    assert enc.target_mean_ == pytest.approx(y_mean)
    X_test = np.array([[1], [0]])
    X_test_trans = enc.transform(X_test)
    assert_allclose(X_test_trans, np.repeat([[y_mean]], 2, axis=0))