from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_object_pairs_hook(self):
    s = '{"xkd":1, "kcw":2, "art":3, "hxm":4, "qrt":5, "pad":6, "hoy":7}'
    p = [('xkd', 1), ('kcw', 2), ('art', 3), ('hxm', 4), ('qrt', 5), ('pad', 6), ('hoy', 7)]
    self.assertEqual(json.loads(s), eval(s))
    self.assertEqual(json.loads(s, object_pairs_hook=lambda x: x), p)
    self.assertEqual(json.load(StringIO(s), object_pairs_hook=lambda x: x), p)
    od = json.loads(s, object_pairs_hook=OrderedDict)
    self.assertEqual(od, OrderedDict(p))
    self.assertEqual(type(od), OrderedDict)
    self.assertEqual(json.loads(s, object_pairs_hook=OrderedDict, object_hook=lambda x: None), OrderedDict(p))