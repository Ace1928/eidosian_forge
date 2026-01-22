from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_spacing():
    rng = np.random.default_rng(0)
    img = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], float)
    result_non_spaced = np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], int)
    result_spaced = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], int)
    img += 0.1 * rng.normal(size=img.shape)
    seg_non_spaced = slic(img, n_segments=2, sigma=0, channel_axis=None, compactness=1.0, start_label=0)
    seg_spaced = slic(img, n_segments=2, sigma=0, spacing=[500, 1], compactness=1.0, channel_axis=None, start_label=0)
    assert_equal(seg_non_spaced, result_non_spaced)
    assert_equal(seg_spaced, result_spaced)