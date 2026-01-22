import numpy as np
from skimage.util import regular_grid
from skimage._shared.testing import assert_equal
def test_regular_grid_full():
    ar = np.zeros((2, 2))
    g = regular_grid(ar, 25)
    assert_equal(g, [slice(None, None, None), slice(None, None, None)])
    ar[g] = 1
    assert_equal(ar.size, ar.sum())