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
def test_basic_nonsvd(self):
    A = array([[1.0, 0, 1], [0, -2.0, 0], [0, 0, 3.0]])
    assert_almost_equal(linalg.cond(A, inf), 4)
    assert_almost_equal(linalg.cond(A, -inf), 2 / 3)
    assert_almost_equal(linalg.cond(A, 1), 4)
    assert_almost_equal(linalg.cond(A, -1), 0.5)
    assert_almost_equal(linalg.cond(A, 'fro'), np.sqrt(265 / 12))