from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_612074(self):
    pat = '[' + regex.escape('‹') + ']'
    self.assertEqual(regex.compile(pat) and 1, 1)