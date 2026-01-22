import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_same_input_shapes():
    data = [(), (1,), (3,), (0, 1), (0, 3), (1, 0), (3, 0), (1, 3), (3, 1), (3, 3)]
    for shape in data:
        input_shapes = [shape]
        assert_shapes_correct(input_shapes, shape)
        input_shapes2 = [shape, shape]
        assert_shapes_correct(input_shapes2, shape)
        input_shapes3 = [shape, shape, shape]
        assert_shapes_correct(input_shapes3, shape)