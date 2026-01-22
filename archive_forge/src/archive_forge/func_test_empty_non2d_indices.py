import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_empty_non2d_indices(self):
    image = np.zeros((10, 10, 10))
    result = peak.peak_local_max(image, footprint=np.ones((3, 3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
    assert result.shape == (0, image.ndim)