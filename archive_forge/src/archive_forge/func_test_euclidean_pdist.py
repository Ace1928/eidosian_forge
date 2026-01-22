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
def test_euclidean_pdist(self):
    a = np.arange(12, dtype=float).reshape(4, 3)
    out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
    umt.euclidean_pdist(a, out)
    b = np.sqrt(np.sum((a[:, None] - a) ** 2, axis=-1))
    b = b[~np.tri(a.shape[0], dtype=bool)]
    assert_almost_equal(out, b)
    assert_raises(ValueError, umt.euclidean_pdist, a)