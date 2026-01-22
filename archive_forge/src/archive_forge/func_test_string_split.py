import os
import itertools
import numpy as np
import imageio.v3 as iio3
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose, fetch
import pytest
def test_string_split():
    test_string = 'z23a'
    test_str_result = ['z', 23, 'a']
    assert_equal(alphanumeric_key(test_string), test_str_result)