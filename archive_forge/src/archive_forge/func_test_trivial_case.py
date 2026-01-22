import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_trivial_case(self):
    trivial = np.zeros((25, 25))
    peak_indices = peak.peak_local_max(trivial, min_distance=1)
    assert type(peak_indices) is np.ndarray
    assert peak_indices.size == 0