from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_117612(self):
    self.assertEqual(regex.findall('(a|(b))', 'aba'), [('a', ''), ('b', 'b'), ('a', '')])