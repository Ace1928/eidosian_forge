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
def test_sum_stability(self):
    a = np.ones(500, dtype=np.float32)
    assert_almost_equal((a / 10.0).sum() - a.size / 10.0, 0, 4)
    a = np.ones(500, dtype=np.float64)
    assert_almost_equal((a / 10.0).sum() - a.size / 10.0, 0, 13)