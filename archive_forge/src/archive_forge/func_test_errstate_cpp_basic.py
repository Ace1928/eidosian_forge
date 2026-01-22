import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_errstate_cpp_basic():
    olderr = sc.geterr()
    with sc.errstate(underflow='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.wrightomega(-1000)
    assert_equal(olderr, sc.geterr())