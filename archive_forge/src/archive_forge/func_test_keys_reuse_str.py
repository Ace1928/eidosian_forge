from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_keys_reuse_str(self):
    s = u'[{"a_key": 1, "b_é": 2}, {"a_key": 3, "b_é": 4}]'.encode('utf8')
    self.check_keys_reuse(s, json.loads)