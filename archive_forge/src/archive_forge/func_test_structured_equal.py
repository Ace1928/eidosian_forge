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
def test_structured_equal(self):

    class MyA(np.ndarray):

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return getattr(ufunc, method)(*(input.view(np.ndarray) for input in inputs), **kwargs)
    a = np.arange(12.0).reshape(4, 3)
    ra = a.view(dtype='f8,f8,f8').squeeze()
    mra = ra.view(MyA)
    target = np.array([True, False, False, False], dtype=bool)
    assert_equal(np.all(target == (mra == ra[0])), True)