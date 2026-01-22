from typing import Callable
import numpy as np
from numpy.testing import assert_array_equal, assert_, suppress_warnings
import pytest
import scipy.special as sc
@pytest.mark.parametrize('func', UFUNCS, ids=UFUNC_NAMES)
def test_nan_inputs(func):
    args = (np.nan,) * func.nin
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'floating point number truncated to an integer')
        try:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                res = func(*args)
        except TypeError:
            return
    if func in POSTPROCESSING:
        res = POSTPROCESSING[func](*res)
    msg = f'got {res} instead of nan'
    assert_array_equal(np.isnan(res), True, err_msg=msg)