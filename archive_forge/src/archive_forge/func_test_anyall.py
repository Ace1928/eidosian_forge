from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_anyall(self):
    self.assertEqual(regex.match('a.b', 'a\nb', regex.DOTALL)[0], 'a\nb')
    self.assertEqual(regex.match('a.*b', 'a\n\nb', regex.DOTALL)[0], 'a\n\nb')