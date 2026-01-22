from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_split(self):
    self.assertEqual(regex.split(':', ':a:b::c'), ['', 'a', 'b', '', 'c'])
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split(':*', ':a:b::c'), ['', '', 'a', '', 'b', '', 'c', ''])
        self.assertEqual(regex.split('(:*)', ':a:b::c'), ['', ':', '', '', 'a', ':', '', '', 'b', '::', '', '', 'c', '', ''])
        self.assertEqual(regex.split('(?::*)', ':a:b::c'), ['', '', 'a', '', 'b', '', 'c', ''])
        self.assertEqual(regex.split('(:)*', ':a:b::c'), ['', ':', '', None, 'a', ':', '', None, 'b', ':', '', None, 'c', None, ''])
    else:
        self.assertEqual(regex.split(':*', ':a:b::c'), ['', 'a', 'b', 'c'])
        self.assertEqual(regex.split('(:*)', ':a:b::c'), ['', ':', 'a', ':', 'b', '::', 'c'])
        self.assertEqual(regex.split('(?::*)', ':a:b::c'), ['', 'a', 'b', 'c'])
        self.assertEqual(regex.split('(:)*', ':a:b::c'), ['', ':', 'a', ':', 'b', ':', 'c'])
    self.assertEqual(regex.split('([b:]+)', ':a:b::c'), ['', ':', 'a', ':b::', 'c'])
    self.assertEqual(regex.split('(b)|(:+)', ':a:b::c'), ['', None, ':', 'a', None, ':', '', 'b', None, '', None, '::', 'c'])
    self.assertEqual(regex.split('(?:b)|(?::+)', ':a:b::c'), ['', 'a', '', '', 'c'])
    self.assertEqual(regex.split('x', 'xaxbxc'), ['', 'a', 'b', 'c'])
    self.assertEqual([m for m in regex.splititer('x', 'xaxbxc')], ['', 'a', 'b', 'c'])
    self.assertEqual(regex.split('(?r)x', 'xaxbxc'), ['c', 'b', 'a', ''])
    self.assertEqual([m for m in regex.splititer('(?r)x', 'xaxbxc')], ['c', 'b', 'a', ''])
    self.assertEqual(regex.split('(x)|(y)', 'xaxbxc'), ['', 'x', None, 'a', 'x', None, 'b', 'x', None, 'c'])
    self.assertEqual([m for m in regex.splititer('(x)|(y)', 'xaxbxc')], ['', 'x', None, 'a', 'x', None, 'b', 'x', None, 'c'])
    self.assertEqual(regex.split('(?r)(x)|(y)', 'xaxbxc'), ['c', 'x', None, 'b', 'x', None, 'a', 'x', None, ''])
    self.assertEqual([m for m in regex.splititer('(?r)(x)|(y)', 'xaxbxc')], ['c', 'x', None, 'b', 'x', None, 'a', 'x', None, ''])
    self.assertEqual(regex.split('(?V1)\\b', 'a b c'), ['', 'a', ' ', 'b', ' ', 'c', ''])
    self.assertEqual(regex.split('(?V1)\\m', 'a b c'), ['', 'a ', 'b ', 'c'])
    self.assertEqual(regex.split('(?V1)\\M', 'a b c'), ['a', ' b', ' c', ''])