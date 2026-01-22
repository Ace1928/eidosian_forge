import sys
import unittest
import Cython.Compiler.StringEncoding as StringEncoding
def test_string_contains_surrogates(self):
    self.assertFalse(StringEncoding.string_contains_surrogates(u'abc'))
    self.assertFalse(StringEncoding.string_contains_surrogates(u'ꯍ'))
    self.assertFalse(StringEncoding.string_contains_surrogates(u'☃'))
    self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800'))
    self.assertTrue(StringEncoding.string_contains_surrogates(u'\udfff'))
    self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800\udfff'))
    self.assertTrue(StringEncoding.string_contains_surrogates(u'\udfff\ud800'))
    self.assertTrue(StringEncoding.string_contains_surrogates(u'\ud800x\udfff'))