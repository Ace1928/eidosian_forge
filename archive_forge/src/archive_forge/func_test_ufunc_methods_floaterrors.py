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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('method', [np.add.accumulate, np.add.reduce, pytest.param(lambda x: np.add.reduceat(x, [0]), id='reduceat'), pytest.param(lambda x: np.log.at(x, [2]), id='at')])
def test_ufunc_methods_floaterrors(method):
    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all='warn'):
        with pytest.warns(RuntimeWarning, match='invalid value'):
            method(arr)
    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            method(arr)