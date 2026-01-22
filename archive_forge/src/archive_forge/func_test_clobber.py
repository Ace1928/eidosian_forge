import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_clobber():
    for func_input_type in img_funcs:
        for func_output_type in img_funcs:
            img = np.random.rand(5, 5)
            img_in = func_input_type(img)
            img_in_before = img_in.copy()
            func_output_type(img_in)
            assert_equal(img_in, img_in_before)