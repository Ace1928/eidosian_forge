from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_bin(self):
    assert_equal(type(self.encode_decode(b'foo')), bytes)