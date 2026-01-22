import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_peak_at_border(self):
    image = np.full((10, 10), -2)
    image[2, 4] = -1
    image[3, 0] = -1
    peaks = peak.peak_local_max(image, min_distance=3)
    assert peaks.size == 0
    peaks = peak.peak_local_max(image, min_distance=3, exclude_border=0)
    assert len(peaks) == 2
    assert [2, 4] in peaks
    assert [3, 0] in peaks