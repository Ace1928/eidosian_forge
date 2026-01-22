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
@pytest.mark.parametrize('bad_offset', [0, int(np.BUFSIZE * 1.5)])
def test_ufunc_input_floatingpoint_error(bad_offset):
    value = 123
    arr = np.array([value] * bad_offset + [np.nan] + [value] * int(1.5 * np.BUFSIZE))
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError):
        np.add(arr, arr, dtype=np.intp, casting='unsafe')