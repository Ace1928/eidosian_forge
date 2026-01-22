from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_line_ending(self):
    self.assertEqual(regex.findall('\\R', '\r\n\n\x0b\x0c\r\x85\u2028\u2029'), ['\r\n', '\n', '\x0b', '\x0c', '\r', '\x85', '\u2028', '\u2029'])
    self.assertEqual(regex.findall(b'\\R', b'\r\n\n\x0b\x0c\r\x85'), [b'\r\n', b'\n', b'\x0b', b'\x0c', b'\r'])