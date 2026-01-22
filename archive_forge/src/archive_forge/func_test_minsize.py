import numpy as np
from skimage import data
from skimage.segmentation import felzenszwalb
from skimage._shared import testing
from skimage._shared.testing import (
def test_minsize():
    img = data.coins()[20:168, 0:128]
    for min_size in np.arange(10, 100, 10):
        segments = felzenszwalb(img, min_size=min_size, sigma=3)
        counts = np.bincount(segments.ravel())
        assert_greater(counts.min() + 1, min_size)
    coffee = data.coffee()[::4, ::4]
    for min_size in np.arange(10, 100, 10):
        segments = felzenszwalb(coffee, min_size=min_size, sigma=3)
        counts = np.bincount(segments.ravel())
        assert_greater(counts.min() + 1, min_size)