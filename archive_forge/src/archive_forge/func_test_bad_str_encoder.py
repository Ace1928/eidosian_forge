from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_bad_str_encoder(self):
    import decimal

    def bad_encoder1(*args):
        return None
    enc = encoder.c_make_encoder(None, lambda obj: str(obj), bad_encoder1, None, ': ', ', ', False, False, False, {}, False, False, False, None, None, 'utf-8', False, False, decimal.Decimal, False)
    self.assertRaises(TypeError, enc, 'spam', 4)
    self.assertRaises(TypeError, enc, {'spam': 42}, 4)

    def bad_encoder2(*args):
        1 / 0
    enc = encoder.c_make_encoder(None, lambda obj: str(obj), bad_encoder2, None, ': ', ', ', False, False, False, {}, False, False, False, None, None, 'utf-8', False, False, decimal.Decimal, False)
    self.assertRaises(ZeroDivisionError, enc, 'spam', 4)