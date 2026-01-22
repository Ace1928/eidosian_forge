from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_spacing_mask():
    rng = np.random.default_rng(0)
    msk = np.zeros((2, 5))
    msk[:, 1:-1] = 1
    img = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], float)
    result_non_spaced = np.array([[0, 1, 1, 2, 0], [0, 1, 2, 2, 0]], int)
    result_spaced = np.array([[0, 1, 1, 1, 0], [0, 2, 2, 2, 0]], int)
    img += 0.1 * rng.normal(size=img.shape)
    seg_non_spaced = slic(img, n_segments=2, sigma=0, channel_axis=None, compactness=1.0, mask=msk)
    seg_spaced = slic(img, n_segments=2, sigma=0, spacing=[50, 1], compactness=1.0, channel_axis=None, mask=msk)
    assert_equal(seg_non_spaced, result_non_spaced)
    assert_equal(seg_spaced, result_spaced)