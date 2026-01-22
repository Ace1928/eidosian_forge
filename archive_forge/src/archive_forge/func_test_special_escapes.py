from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_special_escapes(self):
    self.assertEqual(regex.search('\\b(b.)\\b', 'abcd abc bcd bx')[1], 'bx')
    self.assertEqual(regex.search('\\B(b.)\\B', 'abc bcd bc abxd')[1], 'bx')
    self.assertEqual(regex.search(b'\\b(b.)\\b', b'abcd abc bcd bx', regex.LOCALE)[1], b'bx')
    self.assertEqual(regex.search(b'\\B(b.)\\B', b'abc bcd bc abxd', regex.LOCALE)[1], b'bx')
    self.assertEqual(regex.search('\\b(b.)\\b', 'abcd abc bcd bx', regex.UNICODE)[1], 'bx')
    self.assertEqual(regex.search('\\B(b.)\\B', 'abc bcd bc abxd', regex.UNICODE)[1], 'bx')
    self.assertEqual(regex.search('^abc$', '\nabc\n', regex.M)[0], 'abc')
    self.assertEqual(regex.search('^\\Aabc\\Z$', 'abc', regex.M)[0], 'abc')
    self.assertEqual(regex.search('^\\Aabc\\Z$', '\nabc\n', regex.M), None)
    self.assertEqual(regex.search(b'\\b(b.)\\b', b'abcd abc bcd bx')[1], b'bx')
    self.assertEqual(regex.search(b'\\B(b.)\\B', b'abc bcd bc abxd')[1], b'bx')
    self.assertEqual(regex.search(b'^abc$', b'\nabc\n', regex.M)[0], b'abc')
    self.assertEqual(regex.search(b'^\\Aabc\\Z$', b'abc', regex.M)[0], b'abc')
    self.assertEqual(regex.search(b'^\\Aabc\\Z$', b'\nabc\n', regex.M), None)
    self.assertEqual(regex.search('\\d\\D\\w\\W\\s\\S', '1aa! a')[0], '1aa! a')
    self.assertEqual(regex.search(b'\\d\\D\\w\\W\\s\\S', b'1aa! a', regex.LOCALE)[0], b'1aa! a')
    self.assertEqual(regex.search('\\d\\D\\w\\W\\s\\S', '1aa! a', regex.UNICODE)[0], '1aa! a')