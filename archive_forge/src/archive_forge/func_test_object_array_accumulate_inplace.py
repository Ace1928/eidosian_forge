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
def test_object_array_accumulate_inplace(self):
    arr = np.ones(4, dtype=object)
    arr[:] = [[1] for i in range(4)]
    np.add.accumulate(arr, out=arr)
    np.add.accumulate(arr, out=arr)
    assert_array_equal(arr, np.array([[1] * i for i in [1, 3, 6, 10]], dtype=object))
    arr = np.ones((2, 4), dtype=object)
    arr[0, :] = [[2] for i in range(4)]
    np.add.accumulate(arr, out=arr, axis=-1)
    np.add.accumulate(arr, out=arr, axis=-1)
    assert_array_equal(arr[0, :], np.array([[2] * i for i in [1, 3, 6, 10]], dtype=object))