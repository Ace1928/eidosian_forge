import math
from unittest import TestCase
from simplejson.compat import long_type, text_type
import simplejson as json
from simplejson.decoder import NaN, PosInf, NegInf
def test_degenerates_deny(self):
    for f in (PosInf, NegInf, NaN):
        self.assertRaises(ValueError, json.dumps, f, allow_nan=False)
    for s in ('Infinity', '-Infinity', 'NaN'):
        self.assertRaises(ValueError, json.loads, s, allow_nan=False)
        self.assertRaises(ValueError, json.loads, s)