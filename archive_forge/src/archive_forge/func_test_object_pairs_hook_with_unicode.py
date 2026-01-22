import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_object_pairs_hook_with_unicode(self):
    s = u'{"xkd":1, "kcw":2, "art":3, "hxm":4, "qrt":5, "pad":6, "hoy":7}'
    p = [(u'xkd', 1), (u'kcw', 2), (u'art', 3), (u'hxm', 4), (u'qrt', 5), (u'pad', 6), (u'hoy', 7)]
    self.assertEqual(json.loads(s), eval(s))
    self.assertEqual(json.loads(s, object_pairs_hook=lambda x: x), p)
    od = json.loads(s, object_pairs_hook=json.OrderedDict)
    self.assertEqual(od, json.OrderedDict(p))
    self.assertEqual(type(od), json.OrderedDict)
    self.assertEqual(json.loads(s, object_pairs_hook=json.OrderedDict, object_hook=lambda x: None), json.OrderedDict(p))