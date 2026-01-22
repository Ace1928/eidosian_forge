from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_non_consuming(self):
    self.assertEqual(regex.match('(a(?=\\s[^a]))', 'a b')[1], 'a')
    self.assertEqual(regex.match('(a(?=\\s[^a]*))', 'a b')[1], 'a')
    self.assertEqual(regex.match('(a(?=\\s[abc]))', 'a b')[1], 'a')
    self.assertEqual(regex.match('(a(?=\\s[abc]*))', 'a bc')[1], 'a')
    self.assertEqual(regex.match('(a)(?=\\s\\1)', 'a a')[1], 'a')
    self.assertEqual(regex.match('(a)(?=\\s\\1*)', 'a aa')[1], 'a')
    self.assertEqual(regex.match('(a)(?=\\s(abc|a))', 'a a')[1], 'a')
    self.assertEqual(regex.match('(a(?!\\s[^a]))', 'a a')[1], 'a')
    self.assertEqual(regex.match('(a(?!\\s[abc]))', 'a d')[1], 'a')
    self.assertEqual(regex.match('(a)(?!\\s\\1)', 'a b')[1], 'a')
    self.assertEqual(regex.match('(a)(?!\\s(abc|a))', 'a b')[1], 'a')