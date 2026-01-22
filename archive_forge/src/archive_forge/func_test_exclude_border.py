import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
@pytest.mark.parametrize(['indices'], [[indices] for indices in itertools.product(range(5), range(5))])
def test_exclude_border(indices):
    image = np.zeros((5, 5))
    image[indices] = 1
    assert len(peak.peak_local_max(image, exclude_border=False)) == 1
    assert len(peak.peak_local_max(image, exclude_border=0)) == 1
    if indices[0] in (0, 4) or indices[1] in (0, 4):
        expected_peaks = 0
    else:
        expected_peaks = 1
    assert len(peak.peak_local_max(image, min_distance=1, exclude_border=True)) == expected_peaks
    if indices[0] in (0, 4):
        expected_peaks = 0
    else:
        expected_peaks = 1
    assert len(peak.peak_local_max(image, exclude_border=(1, 0))) == expected_peaks
    if indices[1] in (0, 4):
        expected_peaks = 0
    else:
        expected_peaks = 1
    assert len(peak.peak_local_max(image, exclude_border=(0, 1))) == expected_peaks