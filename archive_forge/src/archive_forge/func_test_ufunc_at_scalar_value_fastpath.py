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
@pytest.mark.parametrize('value', [np.ones(1), np.ones(()), np.float64(1.0), 1.0])
def test_ufunc_at_scalar_value_fastpath(self, value):
    arr = np.zeros(1000)
    index = np.repeat(np.arange(1000), 2)
    np.add.at(arr, index, value)
    assert_array_equal(arr, np.full_like(arr, 2 * value))