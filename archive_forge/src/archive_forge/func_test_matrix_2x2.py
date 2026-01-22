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
def test_matrix_2x2(self):
    A = self.array([[1, 3], [5, 7]], dtype=self.dt)
    assert_almost_equal(norm(A), 84 ** 0.5)
    assert_almost_equal(norm(A, 'fro'), 84 ** 0.5)
    assert_almost_equal(norm(A, 'nuc'), 10.0)
    assert_almost_equal(norm(A, inf), 12.0)
    assert_almost_equal(norm(A, -inf), 4.0)
    assert_almost_equal(norm(A, 1), 10.0)
    assert_almost_equal(norm(A, -1), 6.0)
    assert_almost_equal(norm(A, 2), 9.123105625617661)
    assert_almost_equal(norm(A, -2), 0.8768943743823404)
    assert_raises(ValueError, norm, A, 'nofro')
    assert_raises(ValueError, norm, A, -3)
    assert_raises(ValueError, norm, A, 0)