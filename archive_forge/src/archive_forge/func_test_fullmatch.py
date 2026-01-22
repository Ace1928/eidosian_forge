from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_fullmatch(self):
    self.assertEqual(bool(regex.fullmatch('abc', 'abc')), True)
    self.assertEqual(bool(regex.fullmatch('abc', 'abcx')), False)
    self.assertEqual(bool(regex.fullmatch('abc', 'abcx', endpos=3)), True)
    self.assertEqual(bool(regex.fullmatch('abc', 'xabc', pos=1)), True)
    self.assertEqual(bool(regex.fullmatch('abc', 'xabcy', pos=1)), False)
    self.assertEqual(bool(regex.fullmatch('abc', 'xabcy', pos=1, endpos=4)), True)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'abc')), True)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'abcx')), False)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'abcx', endpos=3)), True)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'xabc', pos=1)), True)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'xabcy', pos=1)), False)
    self.assertEqual(bool(regex.fullmatch('(?r)abc', 'xabcy', pos=1, endpos=4)), True)