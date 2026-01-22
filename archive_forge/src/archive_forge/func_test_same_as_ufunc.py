import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_same_as_ufunc():
    data = [[[(1,), (3,)], (3,)], [[(1, 3), (3, 3)], (3, 3)], [[(3, 1), (3, 3)], (3, 3)], [[(1, 3), (3, 1)], (3, 3)], [[(1, 1), (3, 3)], (3, 3)], [[(1, 1), (1, 3)], (1, 3)], [[(1, 1), (3, 1)], (3, 1)], [[(1, 0), (0, 0)], (0, 0)], [[(0, 1), (0, 0)], (0, 0)], [[(1, 0), (0, 1)], (0, 0)], [[(1, 1), (0, 0)], (0, 0)], [[(1, 1), (1, 0)], (1, 0)], [[(1, 1), (0, 1)], (0, 1)], [[(), (3,)], (3,)], [[(3,), (3, 3)], (3, 3)], [[(3,), (3, 1)], (3, 3)], [[(1,), (3, 3)], (3, 3)], [[(), (3, 3)], (3, 3)], [[(1, 1), (3,)], (1, 3)], [[(1,), (3, 1)], (3, 1)], [[(1,), (1, 3)], (1, 3)], [[(), (1, 3)], (1, 3)], [[(), (3, 1)], (3, 1)], [[(), (0,)], (0,)], [[(0,), (0, 0)], (0, 0)], [[(0,), (0, 1)], (0, 0)], [[(1,), (0, 0)], (0, 0)], [[(), (0, 0)], (0, 0)], [[(1, 1), (0,)], (1, 0)], [[(1,), (0, 1)], (0, 1)], [[(1,), (1, 0)], (1, 0)], [[(), (1, 0)], (1, 0)], [[(), (0, 1)], (0, 1)]]
    for input_shapes, expected_shape in data:
        assert_same_as_ufunc(input_shapes[0], input_shapes[1], 'Shapes: %s %s' % (input_shapes[0], input_shapes[1]))
        assert_same_as_ufunc(input_shapes[1], input_shapes[0])
        assert_same_as_ufunc(input_shapes[0], input_shapes[1], True)
        if () not in input_shapes:
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], False, True)
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], True, True)