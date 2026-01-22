from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_search_star_plus(self):
    self.assertEqual(regex.search('a*', 'xxx').span(0), (0, 0))
    self.assertEqual(regex.search('x*', 'axx').span(), (0, 0))
    self.assertEqual(regex.search('x+', 'axx').span(0), (1, 3))
    self.assertEqual(regex.search('x+', 'axx').span(), (1, 3))
    self.assertEqual(regex.search('x', 'aaa'), None)
    self.assertEqual(regex.match('a*', 'xxx').span(0), (0, 0))
    self.assertEqual(regex.match('a*', 'xxx').span(), (0, 0))
    self.assertEqual(regex.match('x*', 'xxxa').span(0), (0, 3))
    self.assertEqual(regex.match('x*', 'xxxa').span(), (0, 3))
    self.assertEqual(regex.match('a+', 'xxx'), None)