import itertools
import numpy as np
import pytest
from numpy.testing import (
from skimage._shared.testing import expected_warnings
from skimage.color.colorconv import hsv2rgb, rgb2hsv
from skimage.color.colorlabel import label2rgb
@pytest.mark.parametrize('image_type', ['rgb', 'gray', None])
def test_label2rgb_nd(image_type):
    shape = (10, 10)
    if image_type == 'rgb':
        img = np.random.randint(0, 255, shape + (3,), dtype=np.uint8)
    elif image_type == 'gray':
        img = np.random.randint(0, 255, shape, dtype=np.uint8)
    else:
        img = None
    labels = np.zeros(shape, dtype=np.int64)
    labels[2:-2, 1:3] = 1
    labels[3:-3, 6:9] = 2
    labeled_2d = label2rgb(labels, image=img, bg_label=0)
    image_1d = img[5] if image_type is not None else None
    labeled_1d = label2rgb(labels[5], image=image_1d, bg_label=0)
    expected = labeled_2d[5]
    assert_array_equal(labeled_1d, expected)
    image_3d = np.stack((img,) * 4) if image_type is not None else None
    labels_3d = np.stack((labels,) * 4)
    labeled_3d = label2rgb(labels_3d, image=image_3d, bg_label=0)
    for labeled_plane in labeled_3d:
        assert_array_equal(labeled_plane, labeled_2d)