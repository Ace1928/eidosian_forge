from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_lookbehind(self):
    self.assertEqual(regex.search('123(?<=a\\d+)', 'a123').span(), (1, 4))
    self.assertEqual(regex.search('123(?<=a\\d+)', 'b123'), None)
    self.assertEqual(regex.search('123(?<!a\\d+)', 'a123'), None)
    self.assertEqual(regex.search('123(?<!a\\d+)', 'b123').span(), (1, 4))
    self.assertEqual(bool(regex.match('(a)b(?<=b)(c)', 'abc')), True)
    self.assertEqual(regex.match('(a)b(?<=c)(c)', 'abc'), None)
    self.assertEqual(bool(regex.match('(a)b(?=c)(c)', 'abc')), True)
    self.assertEqual(regex.match('(a)b(?=b)(c)', 'abc'), None)
    self.assertEqual(regex.match('(?:(a)|(x))b(?<=(?(2)x|c))c', 'abc'), None)
    self.assertEqual(regex.match('(?:(a)|(x))b(?<=(?(2)b|x))c', 'abc'), None)
    self.assertEqual(bool(regex.match('(?:(a)|(x))b(?<=(?(2)x|b))c', 'abc')), True)
    self.assertEqual(regex.match('(?:(a)|(x))b(?<=(?(1)c|x))c', 'abc'), None)
    self.assertEqual(bool(regex.match('(?:(a)|(x))b(?<=(?(1)b|x))c', 'abc')), True)
    self.assertEqual(bool(regex.match('(?:(a)|(x))b(?=(?(2)x|c))c', 'abc')), True)
    self.assertEqual(regex.match('(?:(a)|(x))b(?=(?(2)c|x))c', 'abc'), None)
    self.assertEqual(bool(regex.match('(?:(a)|(x))b(?=(?(2)x|c))c', 'abc')), True)
    self.assertEqual(regex.match('(?:(a)|(x))b(?=(?(1)b|x))c', 'abc'), None)
    self.assertEqual(bool(regex.match('(?:(a)|(x))b(?=(?(1)c|x))c', 'abc')), True)
    self.assertEqual(regex.match('(a)b(?<=(?(2)x|c))(c)', 'abc'), None)
    self.assertEqual(regex.match('(a)b(?<=(?(2)b|x))(c)', 'abc'), None)
    self.assertEqual(regex.match('(a)b(?<=(?(1)c|x))(c)', 'abc'), None)
    self.assertEqual(bool(regex.match('(a)b(?<=(?(1)b|x))(c)', 'abc')), True)
    self.assertEqual(bool(regex.match('(a)b(?=(?(2)x|c))(c)', 'abc')), True)
    self.assertEqual(regex.match('(a)b(?=(?(2)b|x))(c)', 'abc'), None)
    self.assertEqual(bool(regex.match('(a)b(?=(?(1)c|x))(c)', 'abc')), True)
    self.assertEqual(repr(type(regex.compile('(a)\\2(b)'))), self.PATTERN_CLASS)