from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_hasmethods():
    """Check the HasMethods constraint."""
    constraint = HasMethods(['a', 'b'])

    class _Good:

        def a(self):
            pass

        def b(self):
            pass

    class _Bad:

        def a(self):
            pass
    assert constraint.is_satisfied_by(_Good())
    assert not constraint.is_satisfied_by(_Bad())
    assert str(constraint) == "an object implementing 'a' and 'b'"