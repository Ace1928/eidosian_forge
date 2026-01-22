import sys
from unittest import TestCase
import simplejson as json
import simplejson.decoder
from simplejson.compat import b, PY3
def test_c_scanstring(self):
    if not simplejson.decoder.c_scanstring:
        return
    self._test_scanstring(simplejson.decoder.c_scanstring)
    self.assertTrue(isinstance(simplejson.decoder.c_scanstring('""', 0)[0], str))