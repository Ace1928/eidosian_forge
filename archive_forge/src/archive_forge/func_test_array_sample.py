import re
import unittest
from wsme import exc
from wsme import types
def test_array_sample(self):
    s = types.ArrayType(str).sample()
    assert isinstance(s, list)
    assert s
    assert s[0] == ''