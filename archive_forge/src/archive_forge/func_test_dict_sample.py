import re
import unittest
from wsme import exc
from wsme import types
def test_dict_sample(self):
    s = types.DictType(str, str).sample()
    assert isinstance(s, dict)
    assert s
    assert s == {'': ''}