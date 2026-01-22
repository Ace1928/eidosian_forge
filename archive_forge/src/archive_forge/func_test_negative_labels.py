import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
def test_negative_labels():
    labels = np.array([0, -1, -2, 0])
    rout = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    assert_array_almost_equal(rout, label2rgb(labels, bg_label=0, alpha=1, image_alpha=1))