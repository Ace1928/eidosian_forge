import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_ndarray_exclude_border(self):
    nd_image = np.zeros((5, 5, 5))
    nd_image[[1, 0, 0], [0, 1, 0], [0, 0, 1]] = 1
    nd_image[3, 0, 0] = 1
    nd_image[2, 2, 2] = 1
    expected = np.array([[2, 2, 2]], dtype=int)
    expectedNoBorder = np.array([[0, 0, 1], [2, 2, 2], [3, 0, 0]], dtype=int)
    result = peak.peak_local_max(nd_image, min_distance=2, exclude_border=2)
    assert_array_equal(result, expected)
    assert_array_equal(peak.peak_local_max(nd_image, min_distance=2, exclude_border=2), peak.peak_local_max(nd_image, min_distance=2, exclude_border=True))
    assert_array_equal(peak.peak_local_max(nd_image, min_distance=2, exclude_border=0), peak.peak_local_max(nd_image, min_distance=2, exclude_border=False))
    result = peak.peak_local_max(nd_image, min_distance=2, exclude_border=0)
    assert_array_equal(result, expectedNoBorder)
    peak_idx = peak.peak_local_max(nd_image, exclude_border=False)
    result = np.zeros_like(nd_image, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert_array_equal(result, nd_image.astype(bool))