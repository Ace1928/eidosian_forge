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
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_ufunc_out_casterrors():
    value = 123
    arr = np.array([value] * int(np.BUFSIZE * 1.5) + ['string'] + [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.ones(len(arr), dtype=np.intp)
    count = sys.getrefcount(value)
    with pytest.raises(ValueError):
        np.add(arr, arr, out=out, casting='unsafe')
    assert count == sys.getrefcount(value)
    assert out[-1] == 1
    with pytest.raises(ValueError):
        np.add(arr, arr, out=out, dtype=np.intp, casting='unsafe')
    assert count == sys.getrefcount(value)
    assert out[-1] == 1