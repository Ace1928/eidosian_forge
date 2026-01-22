import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_elementwise_multiply(self):
    A = array([[4, 0, 9], [2, -3, 5]])
    B = array([[0, 7, 0], [0, -4, 0]])
    Asp = self.spcreator(A)
    Bsp = self.spcreator(B)
    assert_almost_equal(Asp.multiply(Bsp).toarray(), A * B)
    assert_almost_equal(Asp.multiply(B).toarray(), A * B)
    C = array([[1 - 2j, 0 + 5j, -1 + 0j], [4 - 3j, -3 + 6j, 5]])
    D = array([[5 + 2j, 7 - 3j, -2 + 1j], [0 - 1j, -4 + 2j, 9]])
    Csp = self.spcreator(C)
    Dsp = self.spcreator(D)
    assert_almost_equal(Csp.multiply(Dsp).toarray(), C * D)
    assert_almost_equal(Csp.multiply(D).toarray(), C * D)
    assert_almost_equal(Asp.multiply(Dsp).toarray(), A * D)
    assert_almost_equal(Asp.multiply(D).toarray(), A * D)