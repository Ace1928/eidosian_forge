import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from skimage.transform import integral_image, integrate
def test_vectorized_integrate():
    r0 = np.array([12, 0, 0, 10, 0, 10, 30])
    c0 = np.array([10, 0, 10, 0, 0, 10, 31])
    r1 = np.array([23, 19, 19, 19, 0, 10, 49])
    c1 = np.array([19, 19, 19, 19, 0, 10, 49])
    expected = np.array([x[12:24, 10:20].sum(), x[:20, :20].sum(), x[:20, 10:20].sum(), x[10:20, :20].sum(), x[0, 0], x[10, 10], x[30:, 31:].sum()])
    start_pts = [(r0[i], c0[i]) for i in range(len(r0))]
    end_pts = [(r1[i], c1[i]) for i in range(len(r0))]
    assert_equal(expected, integrate(s, start_pts, end_pts))