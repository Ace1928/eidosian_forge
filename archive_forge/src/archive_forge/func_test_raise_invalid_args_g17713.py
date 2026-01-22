from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def test_raise_invalid_args_g17713():
    message = 'got an unexpected keyword argument'
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], invalid_arg=True)
    message = ' got multiple values for argument'
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], a=True)
    message = 'missing 1 required positional argument'
    with pytest.raises(TypeError, match=message):
        stats.gmean()
    message = 'takes from 1 to 4 positional arguments but 5 were given'
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], 0, float, [1, 1, 1], 10)