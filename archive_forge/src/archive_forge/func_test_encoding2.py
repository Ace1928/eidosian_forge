import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_encoding2(self):
    u = u'αΩ'
    s = u.encode('utf-8')
    ju = json.dumps(u, encoding='utf-8')
    js = json.dumps(s, encoding='utf-8')
    self.assertEqual(ju, js)