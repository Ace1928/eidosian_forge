from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_list_float_complex(self):
    x = [np.random.rand() + 1j * np.random.rand() for i in range(5)]
    x_rec = self.encode_decode(x)
    assert_array_equal(x, x_rec)
    assert_array_equal([type(e) for e in x], [type(e) for e in x_rec])