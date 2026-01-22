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
@pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
def test_addition_negative_zero(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)
    arr = np.array(neg_zero)
    arr2 = np.array(neg_zero)
    assert _check_neg_zero(arr + arr2)
    arr += arr2
    assert _check_neg_zero(arr)