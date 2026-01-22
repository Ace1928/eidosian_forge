import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
@pytest.mark.parametrize('array_like_input', [False, True])
def test_estimate_affine_3d(array_like_input):
    ndim = 3
    src = np.random.random((25, ndim)) * 2 ** np.arange(7, 7 + ndim)
    matrix = np.array([[4.8, 0.1, 0.2, 25], [0.0, 1.0, 0.1, 30], [0.0, 0.0, 1.0, -2], [0.0, 0.0, 0.0, 1.0]])
    if array_like_input:
        src = [list(c) for c in src]
        matrix = [list(c) for c in matrix]
    tf = AffineTransform(matrix=matrix)
    dst = tf(src)
    dst_noisy = dst + np.random.random((25, ndim))
    if array_like_input:
        dst = [list(c) for c in dst]
    tf2 = AffineTransform(dimensionality=ndim)
    assert tf2.estimate(src, dst_noisy)
    matrix = np.asarray(matrix)
    assert_almost_equal(tf2.params[:, :-1], matrix[:, :-1], decimal=2)
    assert_almost_equal(tf2.params[:, -1], matrix[:, -1], decimal=0)
    _assert_least_squares(tf2, src, dst_noisy)