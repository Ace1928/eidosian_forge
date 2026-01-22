from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_line_boundary(self):
    self.assertEqual(regex.findall('.+', 'Line 1\nLine 2\n'), ['Line 1', 'Line 2'])
    self.assertEqual(regex.findall('.+', 'Line 1\rLine 2\r'), ['Line 1\rLine 2\r'])
    self.assertEqual(regex.findall('.+', 'Line 1\r\nLine 2\r\n'), ['Line 1\r', 'Line 2\r'])
    self.assertEqual(regex.findall('(?w).+', 'Line 1\nLine 2\n'), ['Line 1', 'Line 2'])
    self.assertEqual(regex.findall('(?w).+', 'Line 1\rLine 2\r'), ['Line 1', 'Line 2'])
    self.assertEqual(regex.findall('(?w).+', 'Line 1\r\nLine 2\r\n'), ['Line 1', 'Line 2'])
    self.assertEqual(regex.search('^abc', 'abc').start(), 0)
    self.assertEqual(regex.search('^abc', '\nabc'), None)
    self.assertEqual(regex.search('^abc', '\rabc'), None)
    self.assertEqual(regex.search('(?w)^abc', 'abc').start(), 0)
    self.assertEqual(regex.search('(?w)^abc', '\nabc'), None)
    self.assertEqual(regex.search('(?w)^abc', '\rabc'), None)
    self.assertEqual(regex.search('abc$', 'abc').start(), 0)
    self.assertEqual(regex.search('abc$', 'abc\n').start(), 0)
    self.assertEqual(regex.search('abc$', 'abc\r'), None)
    self.assertEqual(regex.search('(?w)abc$', 'abc').start(), 0)
    self.assertEqual(regex.search('(?w)abc$', 'abc\n').start(), 0)
    self.assertEqual(regex.search('(?w)abc$', 'abc\r').start(), 0)
    self.assertEqual(regex.search('(?m)^abc', 'abc').start(), 0)
    self.assertEqual(regex.search('(?m)^abc', '\nabc').start(), 1)
    self.assertEqual(regex.search('(?m)^abc', '\rabc'), None)
    self.assertEqual(regex.search('(?mw)^abc', 'abc').start(), 0)
    self.assertEqual(regex.search('(?mw)^abc', '\nabc').start(), 1)
    self.assertEqual(regex.search('(?mw)^abc', '\rabc').start(), 1)
    self.assertEqual(regex.search('(?m)abc$', 'abc').start(), 0)
    self.assertEqual(regex.search('(?m)abc$', 'abc\n').start(), 0)
    self.assertEqual(regex.search('(?m)abc$', 'abc\r'), None)
    self.assertEqual(regex.search('(?mw)abc$', 'abc').start(), 0)
    self.assertEqual(regex.search('(?mw)abc$', 'abc\n').start(), 0)
    self.assertEqual(regex.search('(?mw)abc$', 'abc\r').start(), 0)