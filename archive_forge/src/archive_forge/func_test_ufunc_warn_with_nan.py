import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.parametrize('ufunc', [np.sign, np.equal])
def test_ufunc_warn_with_nan(ufunc):
    b = np.array([9218868437227405313], 'i8').view('f8')
    assert np.isnan(b)
    if ufunc.nin == 1:
        ufunc(b)
    elif ufunc.nin == 2:
        ufunc(b, b.copy())
    else:
        raise ValueError('ufunc with more than 2 inputs')