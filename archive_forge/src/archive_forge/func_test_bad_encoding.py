from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_bad_encoding(self):
    with self.assertRaises(UnicodeEncodeError):
        encoder.JSONEncoder(encoding='\udcff').encode({b('key'): 123})