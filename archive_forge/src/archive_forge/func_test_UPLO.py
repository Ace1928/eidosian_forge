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
def test_UPLO(self):
    Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
    Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
    tgt = np.array([-1, 1], dtype=np.double)
    rtol = get_rtol(np.double)
    w, v = np.linalg.eigh(Klo)
    assert_allclose(w, tgt, rtol=rtol)
    w, v = np.linalg.eigh(Klo, UPLO='L')
    assert_allclose(w, tgt, rtol=rtol)
    w, v = np.linalg.eigh(Klo, UPLO='l')
    assert_allclose(w, tgt, rtol=rtol)
    w, v = np.linalg.eigh(Kup, UPLO='U')
    assert_allclose(w, tgt, rtol=rtol)
    w, v = np.linalg.eigh(Kup, UPLO='u')
    assert_allclose(w, tgt, rtol=rtol)