from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_inline_flags(self):
    upper_char = chr(7840)
    lower_char = chr(7841)
    p = regex.compile(upper_char, regex.I | regex.U)
    self.assertEqual(bool(p.match(lower_char)), True)
    p = regex.compile(lower_char, regex.I | regex.U)
    self.assertEqual(bool(p.match(upper_char)), True)
    p = regex.compile('(?i)' + upper_char, regex.U)
    self.assertEqual(bool(p.match(lower_char)), True)
    p = regex.compile('(?i)' + lower_char, regex.U)
    self.assertEqual(bool(p.match(upper_char)), True)
    p = regex.compile('(?iu)' + upper_char)
    self.assertEqual(bool(p.match(lower_char)), True)
    p = regex.compile('(?iu)' + lower_char)
    self.assertEqual(bool(p.match(upper_char)), True)
    self.assertEqual(bool(regex.match('(?i)a', 'A')), True)
    self.assertEqual(regex.match('a(?i)', 'A'), None)