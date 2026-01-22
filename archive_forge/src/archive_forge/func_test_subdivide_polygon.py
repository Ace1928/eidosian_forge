import numpy as np
from skimage.measure import approximate_polygon, subdivide_polygon
from skimage.measure._polygon import _SUBDIVISION_MASKS
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
def test_subdivide_polygon():
    new_square1 = square
    new_square2 = square[:-1]
    new_square3 = square[:-1]
    for _ in range(10):
        square1, square2, square3 = (new_square1, new_square2, new_square3)
        for degree in range(1, 7):
            mask_len = len(_SUBDIVISION_MASKS[degree][0])
            new_square1 = subdivide_polygon(square1, degree)
            assert_array_equal(new_square1[-1], new_square1[0])
            assert_equal(new_square1.shape[0], 2 * square1.shape[0] - 1)
            new_square2 = subdivide_polygon(square2, degree)
            assert_equal(new_square2.shape[0], 2 * (square2.shape[0] - mask_len + 1))
            new_square3 = subdivide_polygon(square3, degree, True)
            assert_equal(new_square3[0], square3[0])
            assert_equal(new_square3[-1], square3[-1])
            assert_equal(new_square3.shape[0], 2 * (square3.shape[0] - mask_len + 2))
    with testing.raises(ValueError):
        subdivide_polygon(square, 0)
    with testing.raises(ValueError):
        subdivide_polygon(square, 8)