from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
def test_unchanged_x0(self):
    x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0
    assert_allclose(self.x00, self.x0)