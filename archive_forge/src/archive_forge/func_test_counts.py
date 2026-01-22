import numpy as np
from skimage.io._plugins._histograms import histograms
def test_counts(self):
    channel = np.arange(255).reshape(51, 5)
    img = np.empty((51, 5, 3), dtype='uint8')
    img[:, :, 0] = channel
    img[:, :, 1] = channel
    img[:, :, 2] = channel
    r, g, b, v = histograms(img, 255)
    np.testing.assert_array_equal(r, g)
    np.testing.assert_array_equal(r, b)
    np.testing.assert_array_equal(r, v)
    np.testing.assert_array_equal(r, np.ones(255))