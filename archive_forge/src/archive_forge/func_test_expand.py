from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_expand(self):
    self.assertEqual(regex.match('(?P<first>first) (?P<second>second)', 'first second').expand('\\2 \\1 \\g<second> \\g<first>'), 'second first second first')