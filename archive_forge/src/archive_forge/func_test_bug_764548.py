from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_764548(self):

    class my_unicode(str):
        pass
    pat = regex.compile(my_unicode('abc'))
    self.assertEqual(pat.match('xyz'), None)