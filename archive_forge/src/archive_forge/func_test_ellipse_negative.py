import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_negative():
    rr, cc = ellipse(-3, -3, 1.7, 1.7)
    rr_, cc_ = np.nonzero(np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]))
    assert_array_equal(rr, rr_ - 5)
    assert_array_equal(cc, cc_ - 5)