from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_decoder_optimizations(self):
    rval = json.loads('{   "key"    :    "value"    ,  "k":"v"    }')
    self.assertEqual(rval, {'key': 'value', 'k': 'v'})