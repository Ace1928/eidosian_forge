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
def test_incompatible_dims(self):
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1, 3.3])
    A = np.vstack([x, np.ones(len(x))]).T
    with assert_raises_regex(LinAlgError, 'Incompatible dimensions'):
        linalg.lstsq(A, y, rcond=None)