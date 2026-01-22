from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_581080(self):
    it = regex.finditer('\\s', 'a b')
    self.assertEqual(next(it).span(), (1, 2))
    self.assertRaises(StopIteration, lambda: next(it))
    scanner = regex.compile('\\s').scanner('a b')
    self.assertEqual(scanner.search().span(), (1, 2))
    self.assertEqual(scanner.search(), None)