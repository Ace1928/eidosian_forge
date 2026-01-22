from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_qualified_re_sub(self):
    self.assertEqual(regex.sub('a', 'b', 'aaaaa'), 'bbbbb')
    self.assertEqual(regex.sub('a', 'b', 'aaaaa', 1), 'baaaa')