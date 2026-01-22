from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_more_segments_than_pixels_mask():
    rng = np.random.default_rng(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.0
    img += 0.0033 * rng.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, sigma=0, n_segments=500, compactness=1, channel_axis=None, convert2lab=False, mask=msk)
    expected = np.arange(seg[2:-2, 2:-2].size) + 1
    assert np.all(seg[2:-2, 2:-2].ravel() == expected)