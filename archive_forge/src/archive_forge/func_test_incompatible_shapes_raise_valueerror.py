import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_incompatible_shapes_raise_valueerror():
    data = [[(3,), (4,)], [(2, 3), (2,)], [(3,), (3,), (4,)], [(1, 3, 4), (2, 3, 3)]]
    for input_shapes in data:
        assert_incompatible_shapes_raise(input_shapes)
        assert_incompatible_shapes_raise(input_shapes[::-1])