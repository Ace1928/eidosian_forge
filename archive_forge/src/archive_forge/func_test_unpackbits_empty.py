import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_unpackbits_empty():
    a = np.empty((0,), dtype=np.uint8)
    b = np.unpackbits(a)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.empty((0,)))