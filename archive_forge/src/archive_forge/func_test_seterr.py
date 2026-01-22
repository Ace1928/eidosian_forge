import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_seterr():
    entry_err = sc.geterr()
    try:
        for category, error_code in _sf_error_code_map.items():
            for action in _sf_error_actions:
                geterr_olderr = sc.geterr()
                seterr_olderr = sc.seterr(**{category: action})
                assert_(geterr_olderr == seterr_olderr)
                newerr = sc.geterr()
                assert_(newerr[category] == action)
                geterr_olderr.pop(category)
                newerr.pop(category)
                assert_(geterr_olderr == newerr)
                _check_action(_sf_error_test_function, (error_code,), action)
    finally:
        sc.seterr(**entry_err)