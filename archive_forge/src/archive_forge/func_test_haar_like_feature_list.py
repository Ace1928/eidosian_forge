from random import shuffle
import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
def test_haar_like_feature_list():
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    haar_list = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
    haar_all = haar_like_feature(img_ii, 0, 0, 5, 5)
    assert_array_equal(haar_list, haar_all)