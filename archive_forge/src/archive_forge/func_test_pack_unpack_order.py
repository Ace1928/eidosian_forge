import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_pack_unpack_order():
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    b_little = np.unpackbits(a, axis=1, bitorder='little')
    b_big = np.unpackbits(a, axis=1, bitorder='big')
    assert_array_equal(b, b_big)
    assert_array_equal(a, np.packbits(b_little, axis=1, bitorder='little'))
    assert_array_equal(b[:, ::-1], b_little)
    assert_array_equal(a, np.packbits(b_big, axis=1, bitorder='big'))
    assert_raises(ValueError, np.unpackbits, a, bitorder='r')
    assert_raises(TypeError, np.unpackbits, a, bitorder=10)