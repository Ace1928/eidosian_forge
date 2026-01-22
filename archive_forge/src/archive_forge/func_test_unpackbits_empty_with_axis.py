import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_unpackbits_empty_with_axis():
    shapes = [([(0,)], (0,)), ([(2, 24, 0), (16, 3, 0), (16, 24, 0)], (16, 24, 0)), ([(2, 0, 24), (16, 0, 24), (16, 0, 3)], (16, 0, 24)), ([(0, 16, 24), (0, 2, 24), (0, 16, 3)], (0, 16, 24)), ([(3, 0, 0), (24, 0, 0), (24, 0, 0)], (24, 0, 0)), ([(0, 24, 0), (0, 3, 0), (0, 24, 0)], (0, 24, 0)), ([(0, 0, 24), (0, 0, 24), (0, 0, 3)], (0, 0, 24)), ([(0, 0, 0), (0, 0, 0), (0, 0, 0)], (0, 0, 0))]
    for in_shapes, out_shape in shapes:
        for ax, in_shape in enumerate(in_shapes):
            a = np.empty(in_shape, dtype=np.uint8)
            b = np.unpackbits(a, axis=ax)
            assert_equal(b.dtype, np.uint8)
            assert_equal(b.shape, out_shape)