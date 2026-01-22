from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_match(self):
    self.assertEqual(regex.match('a', 'a')[:], ('a',))
    self.assertEqual(regex.match('(a)', 'a')[:], ('a', 'a'))
    self.assertEqual(regex.match('(a)', 'a')[0], 'a')
    self.assertEqual(regex.match('(a)', 'a')[1], 'a')
    self.assertEqual(regex.match('(a)', 'a').group(1, 1), ('a', 'a'))
    pat = regex.compile('((a)|(b))(c)?')
    self.assertEqual(pat.match('a')[:], ('a', 'a', 'a', None, None))
    self.assertEqual(pat.match('b')[:], ('b', 'b', None, 'b', None))
    self.assertEqual(pat.match('ac')[:], ('ac', 'a', 'a', None, 'c'))
    self.assertEqual(pat.match('bc')[:], ('bc', 'b', None, 'b', 'c'))
    self.assertEqual(pat.match('bc')[:], ('bc', 'b', None, 'b', 'c'))
    m = regex.match('(a)', 'a')
    self.assertEqual(m.group(), 'a')
    self.assertEqual(m.group(0), 'a')
    self.assertEqual(m.group(1), 'a')
    self.assertEqual(m.group(1, 1), ('a', 'a'))
    pat = regex.compile('(?:(?P<a1>a)|(?P<b2>b))(?P<c3>c)?')
    self.assertEqual(pat.match('a').group(1, 2, 3), ('a', None, None))
    self.assertEqual(pat.match('b').group('a1', 'b2', 'c3'), (None, 'b', None))
    self.assertEqual(pat.match('ac').group(1, 'b2', 3), ('a', None, 'c'))