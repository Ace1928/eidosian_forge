import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_iterate_structure03(self):
    struct = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    out = ndimage.iterate_structure(struct, 2, 1)
    expected = [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]
    assert_array_almost_equal(out[0], expected)
    assert_equal(out[1], [2, 2])