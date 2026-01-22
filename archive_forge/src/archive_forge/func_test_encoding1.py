import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_encoding1(self):
    encoder = json.JSONEncoder(encoding='utf-8')
    u = u'αΩ'
    s = u.encode('utf-8')
    ju = encoder.encode(u)
    js = encoder.encode(s)
    self.assertEqual(ju, js)