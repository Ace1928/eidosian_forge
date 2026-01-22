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
def test_mode_all_but_economic(self):
    a = self.array([[1, 2], [3, 4]])
    b = self.array([[1, 2], [3, 4], [5, 6]])
    for dt in 'fd':
        m1 = a.astype(dt)
        m2 = b.astype(dt)
        self.check_qr(m1)
        self.check_qr(m2)
        self.check_qr(m2.T)
    for dt in 'fd':
        m1 = 1 + 1j * a.astype(dt)
        m2 = 1 + 1j * b.astype(dt)
        self.check_qr(m1)
        self.check_qr(m2)
        self.check_qr(m2.T)