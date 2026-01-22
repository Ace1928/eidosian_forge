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
def test_at_not_none_signature(self):
    a = np.ones((2, 2, 2))
    b = np.ones((1, 2, 2))
    assert_raises(TypeError, np.matmul.at, a, [0], b)
    a = np.array([[[1, 2], [3, 4]]])
    assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])