import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from skimage.transform import integral_image, integrate
def test_integrate_basic():
    assert_equal(x[12:24, 10:20].sum(), integrate(s, (12, 10), (23, 19)))
    assert_equal(x[:20, :20].sum(), integrate(s, (0, 0), (19, 19)))
    assert_equal(x[:20, 10:20].sum(), integrate(s, (0, 10), (19, 19)))
    assert_equal(x[10:20, :20].sum(), integrate(s, (10, 0), (19, 19)))