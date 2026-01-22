from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_more_zerowidth(self):
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split('\\b|:+', 'a::bc'), ['', 'a', '', '', 'bc', ''])
        self.assertEqual(regex.sub('\\b|:+', '-', 'a::bc'), '-a---bc-')
        self.assertEqual(regex.findall('\\b|:+', 'a::bc'), ['', '', '::', '', ''])
        self.assertEqual([m.span() for m in regex.finditer('\\b|:+', 'a::bc')], [(0, 0), (1, 1), (1, 3), (3, 3), (5, 5)])
        self.assertEqual([m.span() for m in regex.finditer('(?m)^\\s*?$', 'foo\n\n\nbar')], [(4, 4), (4, 5), (5, 5)])