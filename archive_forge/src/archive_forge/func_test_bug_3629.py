from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_3629(self):
    self.assertEqual(repr(type(regex.compile('(?P<quote>)(?(quote))'))), self.PATTERN_CLASS)