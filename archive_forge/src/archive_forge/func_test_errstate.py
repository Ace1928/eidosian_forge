import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_errstate():
    for category, error_code in _sf_error_code_map.items():
        for action in _sf_error_actions:
            olderr = sc.geterr()
            with sc.errstate(**{category: action}):
                _check_action(_sf_error_test_function, (error_code,), action)
            assert_equal(olderr, sc.geterr())