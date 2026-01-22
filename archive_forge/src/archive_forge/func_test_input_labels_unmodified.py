import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_input_labels_unmodified(self):
    image = np.zeros((10, 20))
    labels = np.zeros((10, 20), int)
    image[5, 5] = 1
    labels[5, 5] = 3
    labelsin = labels.copy()
    peak.peak_local_max(image, labels=labels, footprint=np.ones((3, 3), bool), min_distance=1, threshold_rel=0, exclude_border=False)
    assert np.all(labels == labelsin)