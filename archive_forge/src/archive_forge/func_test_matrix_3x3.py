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
def test_matrix_3x3(self):
    A = 1 / 10 * self.array([[1, 2, 3], [6, 0, 5], [3, 2, 1]], dtype=self.dt)
    assert_almost_equal(norm(A), 1 / 10 * 89 ** 0.5)
    assert_almost_equal(norm(A, 'fro'), 1 / 10 * 89 ** 0.5)
    assert_almost_equal(norm(A, 'nuc'), 1.3366836911774835)
    assert_almost_equal(norm(A, inf), 1.1)
    assert_almost_equal(norm(A, -inf), 0.6)
    assert_almost_equal(norm(A, 1), 1.0)
    assert_almost_equal(norm(A, -1), 0.4)
    assert_almost_equal(norm(A, 2), 0.8872294032346127)
    assert_almost_equal(norm(A, -2), 0.19456584790481812)