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
@pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
def test_logical_ufuncs_out_cast_check(self, ufunc):
    a = np.array('1')
    c = np.array([1.0, 2.0])
    out = a.copy()
    with pytest.raises(TypeError):
        ufunc(a, c, out=out, casting='equiv')