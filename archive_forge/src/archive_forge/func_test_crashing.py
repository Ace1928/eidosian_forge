import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal, assert_almost_equal, parametrize
from skimage._shared._warnings import expected_warnings
@parametrize('shape', [(100, 100), (5, 8, 13, 17)] * 5)
def test_crashing(shape):
    _test_random(shape)