from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_repeat_minmax(self):
    self.assertEqual(regex.match('^(\\w){1}$', 'abc'), None)
    self.assertEqual(regex.match('^(\\w){1}?$', 'abc'), None)
    self.assertEqual(regex.match('^(\\w){1,2}$', 'abc'), None)
    self.assertEqual(regex.match('^(\\w){1,2}?$', 'abc'), None)
    self.assertEqual(regex.match('^(\\w){3}$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){1,3}$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){1,4}$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){3,4}?$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){3}?$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){1,3}?$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){1,4}?$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^(\\w){3,4}?$', 'abc')[1], 'c')
    self.assertEqual(regex.match('^x{1}$', 'xxx'), None)
    self.assertEqual(regex.match('^x{1}?$', 'xxx'), None)
    self.assertEqual(regex.match('^x{1,2}$', 'xxx'), None)
    self.assertEqual(regex.match('^x{1,2}?$', 'xxx'), None)
    self.assertEqual(regex.match('^x{1}', 'xxx')[0], 'x')
    self.assertEqual(regex.match('^x{1}?', 'xxx')[0], 'x')
    self.assertEqual(regex.match('^x{0,1}', 'xxx')[0], 'x')
    self.assertEqual(regex.match('^x{0,1}?', 'xxx')[0], '')
    self.assertEqual(bool(regex.match('^x{3}$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{1,3}$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{1,4}$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{3,4}?$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{3}?$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{1,3}?$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{1,4}?$', 'xxx')), True)
    self.assertEqual(bool(regex.match('^x{3,4}?$', 'xxx')), True)
    self.assertEqual(regex.match('^x{}$', 'xxx'), None)
    self.assertEqual(bool(regex.match('^x{}$', 'x{}')), True)