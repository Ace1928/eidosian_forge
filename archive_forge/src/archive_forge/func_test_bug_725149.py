from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_725149(self):
    self.assertEqual(regex.match('(a)(?:(?=(b)*)c)*', 'abb')[:], ('a', 'a', None))
    self.assertEqual(regex.match('(a)((?!(b)*))*', 'abb')[:], ('a', 'a', None, None))