from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_527371(self):
    self.assertEqual(regex.match('(a)?a', 'a').lastindex, None)
    self.assertEqual(regex.match('(a)(b)?b', 'ab').lastindex, 1)
    self.assertEqual(regex.match('(?P<a>a)(?P<b>b)?b', 'ab').lastgroup, 'a')
    self.assertEqual(regex.match('(?P<a>a(b))', 'ab').lastgroup, 'a')
    self.assertEqual(regex.match('((a))', 'a').lastindex, 1)