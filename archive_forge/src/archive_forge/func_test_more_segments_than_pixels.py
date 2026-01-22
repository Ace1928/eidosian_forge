from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage import data, filters, img_as_float
from skimage._shared.testing import run_in_parallel, expected_warnings
from skimage.segmentation import slic
def test_more_segments_than_pixels():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.0
    img += 0.0033 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=500, compactness=1, channel_axis=None, convert2lab=False, start_label=0)
    assert np.all(seg.ravel() == np.arange(seg.size))