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
def test_reduction_where_initial_needed(self):
    a = np.arange(9.0).reshape(3, 3)
    m = [False, True, False]
    assert_raises(ValueError, np.maximum.reduce, a, where=m)