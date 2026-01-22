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
def test_ufunc_at_dtypes(self):
    a = np.arange(10)
    np.power.at(a, [1, 2, 3, 2], 3.5)
    assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))