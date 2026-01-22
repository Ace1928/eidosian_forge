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
def test_identityless_reduction_noncontig(self):
    a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
    a = a[1:, 1:, 1:]
    self.check_identityless_reduction(a)