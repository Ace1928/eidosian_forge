import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def test_0_size(self):

    class ArraySubclass(np.ndarray):
        pass
    a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
    res = linalg.cholesky(a)
    assert_equal(a.shape, res.shape)
    assert_(res.dtype.type is np.float64)
    assert_(isinstance(res, np.ndarray))
    a = np.zeros((1, 0, 0), dtype=np.complex64).view(ArraySubclass)
    res = linalg.cholesky(a)
    assert_equal(a.shape, res.shape)
    assert_(res.dtype.type is np.complex64)
    assert_(isinstance(res, np.ndarray))