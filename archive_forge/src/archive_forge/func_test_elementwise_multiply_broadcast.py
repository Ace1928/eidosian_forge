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
def test_elementwise_multiply_broadcast(self):
    A = array([4])
    B = array([[-9]])
    C = array([1, -1, 0])
    D = array([[7, 9, -9]])
    E = array([[3], [2], [1]])
    F = array([[8, 6, 3], [-4, 3, 2], [6, 6, 6]])
    G = [1, 2, 3]
    H = np.ones((3, 4))
    J = H.T
    K = array([[0]])
    L = array([[[1, 2], [0, 1]]])
    Bsp = self.spcreator(B)
    Dsp = self.spcreator(D)
    Esp = self.spcreator(E)
    Fsp = self.spcreator(F)
    Hsp = self.spcreator(H)
    Hspp = self.spcreator(H[0, None])
    Jsp = self.spcreator(J)
    Jspp = self.spcreator(J[:, 0, None])
    Ksp = self.spcreator(K)
    matrices = [A, B, C, D, E, F, G, H, J, K, L]
    spmatrices = [Bsp, Dsp, Esp, Fsp, Hsp, Hspp, Jsp, Jspp, Ksp]
    for i in spmatrices:
        for j in spmatrices:
            try:
                dense_mult = i.toarray() * j.toarray()
            except ValueError:
                assert_raises(ValueError, i.multiply, j)
                continue
            sp_mult = i.multiply(j)
            assert_almost_equal(sp_mult.toarray(), dense_mult)
    for i in spmatrices:
        for j in matrices:
            try:
                dense_mult = i.toarray() * j
            except TypeError:
                continue
            except ValueError:
                assert_raises(ValueError, i.multiply, j)
                continue
            sp_mult = i.multiply(j)
            if issparse(sp_mult):
                assert_almost_equal(sp_mult.toarray(), dense_mult)
            else:
                assert_almost_equal(sp_mult, dense_mult)