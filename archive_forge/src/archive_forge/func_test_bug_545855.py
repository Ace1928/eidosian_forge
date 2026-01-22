from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_545855(self):
    self.assertRaisesRegex(regex.error, self.BAD_SET, lambda: regex.compile('foo[a-'))