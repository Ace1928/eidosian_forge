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
def test_zerosize_reduction(self):
    for a in [[], np.array([], dtype=object)]:
        assert_equal(np.sum(a), 0)
        assert_equal(np.prod(a), 1)
        assert_equal(np.any(a), False)
        assert_equal(np.all(a), True)
        assert_raises(ValueError, np.max, a)
        assert_raises(ValueError, np.min, a)