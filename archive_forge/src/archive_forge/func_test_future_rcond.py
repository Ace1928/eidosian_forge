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
def test_future_rcond(self):
    a = np.array([[0.0, 1.0, 0.0, 1.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 2.0, 3.0, 0.0]]).T
    b = np.array([1, 0, 0, 0, 0, 0])
    with suppress_warnings() as sup:
        w = sup.record(FutureWarning, '`rcond` parameter will change')
        x, residuals, rank, s = linalg.lstsq(a, b)
        assert_(rank == 4)
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=-1)
        assert_(rank == 4)
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        assert_(rank == 3)
        assert_(len(w) == 1)