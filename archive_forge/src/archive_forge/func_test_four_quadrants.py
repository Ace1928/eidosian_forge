import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_four_quadrants(self):
    image = np.random.uniform(size=(20, 30))
    i, j = np.mgrid[0:20, 0:30]
    labels = 1 + (i >= 10) + (j >= 15) * 2
    i, j = np.mgrid[-3:4, -3:4]
    footprint = i * i + j * j <= 9
    expected = np.zeros(image.shape, float)
    for imin, imax in ((0, 10), (10, 20)):
        for jmin, jmax in ((0, 15), (15, 30)):
            expected[imin:imax, jmin:jmax] = ndi.maximum_filter(image[imin:imax, jmin:jmax], footprint=footprint)
    expected = expected == image
    peak_idx = peak.peak_local_max(image, labels=labels, footprint=footprint, min_distance=1, threshold_rel=0, exclude_border=False)
    result = np.zeros_like(image, dtype=bool)
    result[tuple(peak_idx.T)] = True
    assert np.all(result == expected)