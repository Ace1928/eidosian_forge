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
def test_vector_as_last_argument(self):
    A = np.random.random((6, 2))
    B = np.random.random((2, 6))
    C = np.random.random((6, 2))
    D1d = np.random.random(2)
    assert_equal(multi_dot([A, B, C, D1d]).shape, (6,))