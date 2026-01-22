import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(hasattr(np.__config__, 'blas_ssl2_info'), reason='gh-22982')
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('dtype', np.typecodes['Float'])
def test_floor_division_errors(self, dtype):
    fnan = np.array(np.nan, dtype=dtype)
    fone = np.array(1.0, dtype=dtype)
    fzer = np.array(0.0, dtype=dtype)
    finf = np.array(np.inf, dtype=dtype)
    with np.errstate(divide='raise', invalid='ignore'):
        assert_raises(FloatingPointError, np.floor_divide, fone, fzer)
    with np.errstate(divide='ignore', invalid='raise'):
        np.floor_divide(fone, fzer)
    with np.errstate(all='raise'):
        np.floor_divide(fnan, fone)
        np.floor_divide(fone, fnan)
        np.floor_divide(fnan, fzer)
        np.floor_divide(fzer, fnan)