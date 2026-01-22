import re
import unittest
from wsme import exc
from wsme import types
def test_text_attribute_conversion(self):

    class SType(object):
        atext = types.text
        abytes = types.bytes
    types.register_type(SType)
    obj = SType()
    obj.atext = b'somebytes'
    assert obj.atext == 'somebytes'
    assert isinstance(obj.atext, types.text)
    obj.abytes = 'sometext'
    assert obj.abytes == b'sometext'
    assert isinstance(obj.abytes, types.bytes)