import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_affine_shear():
    shear = 0.1
    cx = -np.tan(shear)
    expected = np.array([[1, cx, 0], [0, 1, 0], [0, 0, 1]])
    tform = AffineTransform(shear=shear)
    assert_almost_equal(tform.params, expected)
    shear = (1.2, 0.8)
    cx = -np.tan(shear[0])
    cy = -np.tan(shear[1])
    expected = np.array([[1, cx, 0], [cy, 1, 0], [0, 0, 1]])
    tform = AffineTransform(shear=shear)
    assert_almost_equal(tform.params, expected)