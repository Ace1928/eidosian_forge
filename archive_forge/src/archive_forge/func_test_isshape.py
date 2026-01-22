import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_isshape(self):
    assert_equal(sputils.isshape((1, 2)), True)
    assert_equal(sputils.isshape((5, 2)), True)
    assert_equal(sputils.isshape((1.5, 2)), False)
    assert_equal(sputils.isshape((2, 2, 2)), False)
    assert_equal(sputils.isshape(([2], 2)), False)
    assert_equal(sputils.isshape((-1, 2), nonneg=False), True)
    assert_equal(sputils.isshape((2, -1), nonneg=False), True)
    assert_equal(sputils.isshape((-1, 2), nonneg=True), False)
    assert_equal(sputils.isshape((2, -1), nonneg=True), False)
    assert_equal(sputils.isshape((1.5, 2), allow_ndim=True), False)
    assert_equal(sputils.isshape(([2], 2), allow_ndim=True), False)
    assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_ndim=True), False)
    assert_equal(sputils.isshape((2,), allow_ndim=True), True)
    assert_equal(sputils.isshape((2, 2), allow_ndim=True), True)
    assert_equal(sputils.isshape((2, 2, 2), allow_ndim=True), True)