from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_search_anchor(self):
    self.assertEqual(regex.findall('\\G\\w{2}', 'abcd ef'), ['ab', 'cd'])