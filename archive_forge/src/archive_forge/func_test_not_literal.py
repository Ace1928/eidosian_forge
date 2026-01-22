from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_not_literal(self):
    self.assertEqual(regex.search('\\s([^a])', ' b')[1], 'b')
    self.assertEqual(regex.search('\\s([^a]*)', ' bb')[1], 'bb')