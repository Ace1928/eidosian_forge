import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_label2rgb_shape_errors():
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[2:5, 2:5] = 1
    with pytest.raises(ValueError):
        label2rgb(labels, img[1:])
    with pytest.raises(ValueError):
        label2rgb(labels, img[..., np.newaxis])
    with pytest.raises(ValueError):
        label2rgb(labels, np.concatenate((img, img), axis=-1))