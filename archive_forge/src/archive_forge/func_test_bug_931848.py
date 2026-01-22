from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_931848(self):
    pattern = '[.。．｡]'
    self.assertEqual(regex.compile(pattern).split('a.b.c'), ['a', 'b', 'c'])