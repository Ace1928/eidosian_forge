from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_114660(self):
    self.assertEqual(regex.sub('(\\S)\\s+(\\S)', '\\1 \\2', 'hello  there'), 'hello there')