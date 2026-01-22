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
def test_reducelike_out_promotes(self):
    arr = np.ones(1000, dtype=np.uint8)
    out = np.zeros((), dtype=np.uint16)
    assert np.add.reduce(arr, out=out) == 1000
    arr[:10] = 2
    assert np.multiply.reduce(arr, out=out) == 2 ** 10
    arr = np.full(5, 2 ** 25 - 1, dtype=np.int64)
    res = np.zeros((), dtype=np.float32)
    single_res = np.zeros((), dtype=np.float32)
    np.multiply.reduce(arr, out=single_res, dtype=np.float32)
    assert single_res != res