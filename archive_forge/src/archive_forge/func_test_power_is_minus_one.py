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
def test_power_is_minus_one(self, dt):

    def tz(mat):
        invmat = matrix_power(mat, -1)
        mmul = matmul if mat.dtype != object else dot
        assert_almost_equal(mmul(invmat, mat), identity_like_generalized(mat))
    for mat in self.rshft_all:
        if dt not in self.dtnoinv:
            tz(mat.astype(dt))