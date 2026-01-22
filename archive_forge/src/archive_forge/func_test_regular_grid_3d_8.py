import numpy as np
from skimage.util import regular_grid
from skimage._shared.testing import assert_equal
def test_regular_grid_3d_8():
    ar = np.zeros((3, 20, 40))
    g = regular_grid(ar.shape, 8)
    assert_equal(g, [slice(1.0, None, 3.0), slice(5.0, None, 10.0), slice(5.0, None, 10.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 8)