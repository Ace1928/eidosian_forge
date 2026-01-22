from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_make_encoder(self):
    self.assertRaises(TypeError, encoder.c_make_encoder, None, "Í}=N\x12Lùy×Rº\x82ò'J}\xa0Êu", None)