from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_ascii_and_unicode_flag(self):
    for flags in (0, regex.UNICODE):
        pat = regex.compile('À', flags | regex.IGNORECASE)
        self.assertEqual(bool(pat.match('à')), True)
        pat = regex.compile('\\w', flags)
        self.assertEqual(bool(pat.match('à')), True)
    pat = regex.compile('À', regex.ASCII | regex.IGNORECASE)
    self.assertEqual(pat.match('à'), None)
    pat = regex.compile('(?a)À', regex.IGNORECASE)
    self.assertEqual(pat.match('à'), None)
    pat = regex.compile('\\w', regex.ASCII)
    self.assertEqual(pat.match('à'), None)
    pat = regex.compile('(?a)\\w')
    self.assertEqual(pat.match('à'), None)
    for flags in (0, regex.ASCII):
        pat = regex.compile(b'\xc0', flags | regex.IGNORECASE)
        self.assertEqual(pat.match(b'\xe0'), None)
        pat = regex.compile(b'\\w')
        self.assertEqual(pat.match(b'\xe0'), None)
    self.assertRaisesRegex(ValueError, self.MIXED_FLAGS, lambda: regex.compile('(?au)\\w'))