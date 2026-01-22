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
def test_ufunc_at_negative(self):
    arr = np.ones(5, dtype=np.int32)
    indx = np.arange(5)
    umt.indexed_negative.at(arr, indx)
    assert np.all(arr == [-1, -1, -1, -200, -1])