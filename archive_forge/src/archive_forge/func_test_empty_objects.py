from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_empty_objects(self):
    s = '{}'
    self.assertEqual(json.loads(s), eval(s))
    s = '[]'
    self.assertEqual(json.loads(s), eval(s))
    s = '""'
    self.assertEqual(json.loads(s), eval(s))