import sys
from unittest import TestCase
import simplejson as json
import simplejson.decoder
from simplejson.compat import b, PY3
def test_issue3623(self):
    self.assertRaises(ValueError, json.decoder.scanstring, 'xxx', 1, 'xxx')
    self.assertRaises(UnicodeDecodeError, json.encoder.encode_basestring_ascii, b('xxÿ'))