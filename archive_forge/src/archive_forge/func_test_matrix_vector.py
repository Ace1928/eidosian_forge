from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ..affines import (
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes
def test_matrix_vector():
    for M, N in ((4, 4), (5, 4), (4, 5)):
        xform = np.zeros((M, N))
        xform[:-1, :] = np.random.normal(size=(M - 1, N))
        xform[-1, -1] = 1
        newmat, newvec = to_matvec(xform)
        mat = xform[:-1, :-1]
        vec = xform[:-1, -1]
        assert_array_equal(newmat, mat)
        assert_array_equal(newvec, vec)
        assert newvec.shape == (M - 1,)
        assert_array_equal(from_matvec(mat, vec), xform)
        xform_not = xform[:]
        xform_not[:-1, :] = 0
        assert_array_equal(from_matvec(mat), xform)
        assert_array_equal(from_matvec(mat, None), xform)
    newmat, newvec = to_matvec(xform.tolist())
    assert_array_equal(newmat, mat)
    assert_array_equal(newvec, vec)
    assert_array_equal(from_matvec(mat.tolist(), vec.tolist()), xform)