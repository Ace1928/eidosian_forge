import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_generate_structure03(self):
    struct = ndimage.generate_binary_structure(2, 1)
    assert_array_almost_equal(struct, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])