from unittest import TestCase
import simplejson.encoder
from simplejson.compat import b
def test_c_encode_basestring_ascii(self):
    if not simplejson.encoder.c_encode_basestring_ascii:
        return
    self._test_encode_basestring_ascii(simplejson.encoder.c_encode_basestring_ascii)