import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_many_objects(self):
    mask = np.zeros([500, 500], dtype=bool)
    x, y = np.indices((500, 500))
    x_c = x // 20 * 20 + 10
    y_c = y // 20 * 20 + 10
    mask[(x - x_c) ** 2 + (y - y_c) ** 2 < 8 ** 2] = True
    labels, num_objs = ndi.label(mask)
    dist = ndi.distance_transform_edt(mask)
    local_max = peak.peak_local_max(dist, min_distance=20, exclude_border=False, labels=labels)
    assert len(local_max) == 625