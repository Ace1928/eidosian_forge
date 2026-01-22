from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_possessive(self):
    self.assertEqual(regex.search('a?a', 'a').span(), (0, 1))
    self.assertEqual(regex.search('a*a', 'aaa').span(), (0, 3))
    self.assertEqual(regex.search('a+a', 'aaa').span(), (0, 3))
    self.assertEqual(regex.search('a{1,3}a', 'aaa').span(), (0, 3))
    self.assertEqual(regex.search('(?:ab)?ab', 'ab').span(), (0, 2))
    self.assertEqual(regex.search('(?:ab)*ab', 'ababab').span(), (0, 6))
    self.assertEqual(regex.search('(?:ab)+ab', 'ababab').span(), (0, 6))
    self.assertEqual(regex.search('(?:ab){1,3}ab', 'ababab').span(), (0, 6))
    self.assertEqual(regex.search('a?+a', 'a'), None)
    self.assertEqual(regex.search('a*+a', 'aaa'), None)
    self.assertEqual(regex.search('a++a', 'aaa'), None)
    self.assertEqual(regex.search('a{1,3}+a', 'aaa'), None)
    self.assertEqual(regex.search('(?:ab)?+ab', 'ab'), None)
    self.assertEqual(regex.search('(?:ab)*+ab', 'ababab'), None)
    self.assertEqual(regex.search('(?:ab)++ab', 'ababab'), None)
    self.assertEqual(regex.search('(?:ab){1,3}+ab', 'ababab'), None)