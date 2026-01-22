import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_bad_out_shape(self):
    a = array([1, 2])
    b = array([3, 4])
    assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
    assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
    assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))
    concatenate((a, b), out=np.empty(4))