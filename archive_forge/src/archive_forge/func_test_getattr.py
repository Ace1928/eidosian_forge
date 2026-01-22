from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_getattr(self):
    self.assertEqual(regex.compile('(?i)(a)(b)').pattern, '(?i)(a)(b)')
    self.assertEqual(regex.compile('(?i)(a)(b)').flags, regex.I | regex.U | regex.DEFAULT_VERSION)
    self.assertEqual(regex.compile(b'(?i)(a)(b)').flags, regex.A | regex.I | regex.DEFAULT_VERSION)
    self.assertEqual(regex.compile('(?i)(a)(b)').groups, 2)
    self.assertEqual(regex.compile('(?i)(a)(b)').groupindex, {})
    self.assertEqual(regex.compile('(?i)(?P<first>a)(?P<other>b)').groupindex, {'first': 1, 'other': 2})
    self.assertEqual(regex.match('(a)', 'a').pos, 0)
    self.assertEqual(regex.match('(a)', 'a').endpos, 1)
    self.assertEqual(regex.search('b(c)', 'abcdef').pos, 0)
    self.assertEqual(regex.search('b(c)', 'abcdef').endpos, 6)
    self.assertEqual(regex.search('b(c)', 'abcdef').span(), (1, 3))
    self.assertEqual(regex.search('b(c)', 'abcdef').span(1), (2, 3))
    self.assertEqual(regex.match('(a)', 'a').string, 'a')
    self.assertEqual(regex.match('(a)', 'a').regs, ((0, 1), (0, 1)))
    self.assertEqual(repr(type(regex.match('(a)', 'a').re)), self.PATTERN_CLASS)
    p = regex.compile('abc(?P<n>def)')
    p.groupindex['n'] = 0
    self.assertEqual(p.groupindex['n'], 1)