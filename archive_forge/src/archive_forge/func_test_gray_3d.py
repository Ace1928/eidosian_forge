from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_gray_3d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 22))
    slices = []
    for dim_size in img.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    shades = np.arange(0, 1.000001, 1.0 / 7)
    for s, sh in zip(slices, shades):
        img[s] = sh
    img += 0.001 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=8, compactness=1, channel_axis=None, convert2lab=False, start_label=0)
    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_equal(seg[s], c)