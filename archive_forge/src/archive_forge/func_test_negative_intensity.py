import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_negative_intensity():
    labels = np.arange(100).reshape(10, 10)
    image = np.full((10, 10), -1, dtype='float64')
    assert_warns(UserWarning, label2rgb, labels, image, bg_label=-1)