from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
@pytest.mark.parametrize('interval', [Interval(Real, 0, 1, closed='left'), Interval(Real, None, None, closed='both')])
def test_nan_not_in_interval(interval):
    """Check that np.nan is not in any interval."""
    assert np.nan not in interval