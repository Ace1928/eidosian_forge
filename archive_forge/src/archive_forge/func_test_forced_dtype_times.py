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
def test_forced_dtype_times(self):
    a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
    np.maximum(a, a, dtype='M')
    np.maximum.reduce(a, dtype='M')
    arr = np.arange(10, dtype='m8[s]')
    np.add(arr, arr, dtype='m')
    np.maximum(arr, arr, dtype='m')