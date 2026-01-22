import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_ensure_ascii_linebreak_encoding(self):
    s1 = u'\u2029\u2028'
    s2 = s1.encode('utf8')
    expect = '"\\u2029\\u2028"'
    expect_non_ascii = u'"\u2029\u2028"'
    self.assertEqual(json.dumps(s1), expect)
    self.assertEqual(json.dumps(s2), expect)
    self.assertEqual(json.dumps(s1, ensure_ascii=False), expect_non_ascii)
    self.assertEqual(json.dumps(s2, ensure_ascii=False), expect_non_ascii)