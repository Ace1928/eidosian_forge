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
def test_kwarg_exact(self):
    assert_raises(TypeError, np.add, 1, 2, castingx='safe')
    assert_raises(TypeError, np.add, 1, 2, dtypex=int)
    assert_raises(TypeError, np.add, 1, 2, extobjx=[4096])
    assert_raises(TypeError, np.add, 1, 2, outx=None)
    assert_raises(TypeError, np.add, 1, 2, sigx='ii->i')
    assert_raises(TypeError, np.add, 1, 2, signaturex='ii->i')
    assert_raises(TypeError, np.add, 1, 2, subokx=False)
    assert_raises(TypeError, np.add, 1, 2, wherex=[True])