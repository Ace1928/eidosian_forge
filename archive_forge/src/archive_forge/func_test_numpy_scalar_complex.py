from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_numpy_scalar_complex(self):
    x = np.complex64(np.random.rand() + 1j * np.random.rand())
    x_rec = self.encode_decode(x)
    assert_equal(x, x_rec)
    assert_equal(type(x), type(x_rec))