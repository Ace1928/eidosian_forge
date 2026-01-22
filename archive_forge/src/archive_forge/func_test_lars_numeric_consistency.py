import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
@pytest.mark.parametrize('LARS, has_coef_path, args', ((Lars, True, {}), (LassoLars, True, {}), (LassoLarsIC, False, {}), (LarsCV, True, {}), (LassoLarsCV, True, {'max_iter': 5})))
def test_lars_numeric_consistency(LARS, has_coef_path, args):
    rtol = 1e-05
    atol = 1e-05
    rng = np.random.RandomState(0)
    X_64 = rng.rand(10, 6)
    y_64 = rng.rand(10)
    model_64 = LARS(**args).fit(X_64, y_64)
    model_32 = LARS(**args).fit(X_64.astype(np.float32), y_64.astype(np.float32))
    assert_allclose(model_64.coef_, model_32.coef_, rtol=rtol, atol=atol)
    if has_coef_path:
        assert_allclose(model_64.coef_path_, model_32.coef_path_, rtol=rtol, atol=atol)
    assert_allclose(model_64.intercept_, model_32.intercept_, rtol=rtol, atol=atol)