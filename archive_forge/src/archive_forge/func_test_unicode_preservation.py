import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_unicode_preservation(self):
    self.assertEqual(type(json.loads(u'""')), text_type)
    self.assertEqual(type(json.loads(u'"a"')), text_type)
    self.assertEqual(type(json.loads(u'["a"]')[0]), text_type)