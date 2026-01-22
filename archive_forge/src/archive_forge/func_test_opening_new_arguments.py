import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_opening_new_arguments(self):
    opened_new = ndimage.binary_opening(self.array, self.sq3x3, 1, None, 0, None, 0, False)
    assert_array_equal(opened_new, self.opened_old)