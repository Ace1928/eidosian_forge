import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_2d_negative():
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(5, -5), (6, -6)])
    np.testing.assert_array_equal(out, data[5:-5, 6:-6])