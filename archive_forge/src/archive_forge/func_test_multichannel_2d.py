from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_multichannel_2d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 20, 8))
    img[:10, :10, 0:2] = 1
    img[:10, 10:, 2:4] = 1
    img[10:, :10, 4:6] = 1
    img[10:, 10:, 6:8] = 1
    img += 0.01 * rng.normal(size=img.shape)
    img = np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, enforce_connectivity=False, start_label=0)
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_equal(seg[:10, :10], 0)
    assert_equal(seg[10:, :10], 2)
    assert_equal(seg[:10, 10:], 1)
    assert_equal(seg[10:, 10:], 3)