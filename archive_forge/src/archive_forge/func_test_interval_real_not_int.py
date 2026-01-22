from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_interval_real_not_int():
    """Check for the type RealNotInt in the Interval constraint."""
    constraint = Interval(RealNotInt, 0, 1, closed='both')
    assert constraint.is_satisfied_by(1.0)
    assert not constraint.is_satisfied_by(1)