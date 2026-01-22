from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_no_validation():
    """Check that validation can be skipped for a parameter."""

    @validate_params({'param1': [int, None], 'param2': 'no_validation'}, prefer_skip_nested_validation=True)
    def f(param1=None, param2=None):
        pass
    with pytest.raises(InvalidParameterError, match="The 'param1' parameter"):
        f(param1='wrong')

    class SomeType:
        pass
    f(param2=SomeType)
    f(param2=SomeType())