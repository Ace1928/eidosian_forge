import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from skimage.transform import integral_image, integrate
def test_integrate_single():
    assert_equal(x[0, 0], integrate(s, (0, 0), (0, 0)))
    assert_equal(x[10, 10], integrate(s, (10, 10), (10, 10)))