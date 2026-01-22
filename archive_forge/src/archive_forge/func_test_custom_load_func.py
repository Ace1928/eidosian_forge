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
def test_custom_load_func(self):

    def load_fn(x):
        return x
    ic = ImageCollection(os.pathsep.join(self.pattern), load_func=load_fn)
    assert_equal(ic[0], self.pattern[0])