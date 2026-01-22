from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_dict_float(self):
    x = {b'foo': 1.0, b'bar': 2.0}
    x_rec = self.encode_decode(x)
    assert_array_equal(sorted(x.values()), sorted(x_rec.values()))
    assert_array_equal([type(e) for e in sorted(x.values())], [type(e) for e in sorted(x_rec.values())])
    assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
    assert_array_equal([type(e) for e in sorted(x.keys())], [type(e) for e in sorted(x_rec.keys())])