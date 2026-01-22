from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_interval_inf_in_bounds():
    """Check that inf is included iff a bound is closed and set to None.

    Only valid for real intervals.
    """
    interval = Interval(Real, 0, None, closed='right')
    assert np.inf in interval
    interval = Interval(Real, None, 0, closed='left')
    assert -np.inf in interval
    interval = Interval(Real, None, None, closed='neither')
    assert np.inf not in interval
    assert -np.inf not in interval