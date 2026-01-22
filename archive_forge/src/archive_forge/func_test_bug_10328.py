from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_10328(self):
    pat = regex.compile('(?mV0)(?P<trailing_ws>[ \\t]+\\r*$)|(?P<no_final_newline>(?<=[^\\n])\\Z)')
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(pat.subn(lambda m: '<' + m.lastgroup + '>', 'foobar '), ('foobar<trailing_ws><no_final_newline>', 2))
    else:
        self.assertEqual(pat.subn(lambda m: '<' + m.lastgroup + '>', 'foobar '), ('foobar<trailing_ws>', 1))
    self.assertEqual([m.group() for m in pat.finditer('foobar ')], [' ', ''])
    pat = regex.compile('(?mV1)(?P<trailing_ws>[ \\t]+\\r*$)|(?P<no_final_newline>(?<=[^\\n])\\Z)')
    self.assertEqual(pat.subn(lambda m: '<' + m.lastgroup + '>', 'foobar '), ('foobar<trailing_ws><no_final_newline>', 2))
    self.assertEqual([m.group() for m in pat.finditer('foobar ')], [' ', ''])