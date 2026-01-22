import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_flat_peak(self):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:3, 1:3] = 10
    peaks = peak.peak_local_max(image, min_distance=1)
    assert len(peaks) == 4