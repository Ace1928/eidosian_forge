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
def test_cast_index_fastpath(self):
    arr = np.zeros(10)
    values = np.ones(100000)
    index = np.zeros(len(values), dtype=np.uint8)
    np.add.at(arr, index, values)
    assert arr[0] == len(values)