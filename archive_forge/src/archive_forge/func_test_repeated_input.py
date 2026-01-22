import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_repeated_input(self):
    length_of_vector = 5
    x = np.arange(length_of_vector)
    out = ix_(x, x)
    assert_equal(out[0].shape, (length_of_vector, 1))
    assert_equal(out[1].shape, (1, length_of_vector))
    assert_equal(x.shape, (length_of_vector,))