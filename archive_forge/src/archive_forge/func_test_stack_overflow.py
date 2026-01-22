from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_stack_overflow(self):
    self.assertEqual(regex.match('(x)*', 50000 * 'x')[1], 'x')
    self.assertEqual(regex.match('(x)*y', 50000 * 'x' + 'y')[1], 'x')
    self.assertEqual(regex.match('(x)*?y', 50000 * 'x' + 'y')[1], 'x')