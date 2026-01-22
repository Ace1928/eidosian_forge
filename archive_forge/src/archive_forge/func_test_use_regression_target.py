import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
def test_use_regression_target():
    """Check inferred and specified `target_type` on regression target."""
    X = np.array([[0, 1, 0, 1, 0, 1]]).T
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    enc = TargetEncoder(cv=2)
    with pytest.warns(UserWarning, match=re.escape('The least populated class in y has only 1 members, which is less than n_splits=2.')):
        enc.fit_transform(X, y)
    assert enc.target_type_ == 'multiclass'
    enc = TargetEncoder(cv=2, target_type='continuous')
    enc.fit_transform(X, y)
    assert enc.target_type_ == 'continuous'