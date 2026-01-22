from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_14462(self):
    group_name = 'ÿ'
    self.assertEqual(regex.search('(?P<' + group_name + '>a)', 'abc').group(group_name), 'a')