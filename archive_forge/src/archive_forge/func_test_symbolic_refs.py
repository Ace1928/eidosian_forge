from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_symbolic_refs(self):
    self.assertRaisesRegex(regex.error, self.MISSING_GT, lambda: regex.sub('(?P<a>x)', '\\g<a', 'xx'))
    self.assertRaisesRegex(regex.error, self.MISSING_GROUP_NAME, lambda: regex.sub('(?P<a>x)', '\\g<', 'xx'))
    self.assertRaisesRegex(regex.error, self.MISSING_LT, lambda: regex.sub('(?P<a>x)', '\\g', 'xx'))
    self.assertRaisesRegex(regex.error, self.BAD_GROUP_NAME, lambda: regex.sub('(?P<a>x)', '\\g<a a>', 'xx'))
    self.assertRaisesRegex(regex.error, self.BAD_GROUP_NAME, lambda: regex.sub('(?P<a>x)', '\\g<1a1>', 'xx'))
    self.assertRaisesRegex(IndexError, self.UNKNOWN_GROUP_I, lambda: regex.sub('(?P<a>x)', '\\g<ab>', 'xx'))
    self.assertEqual(regex.sub('(?P<a>x)|(?P<b>y)', '\\g<b>', 'xx'), '')
    self.assertEqual(regex.sub('(?P<a>x)|(?P<b>y)', '\\2', 'xx'), '')
    self.assertRaisesRegex(regex.error, self.BAD_GROUP_NAME, lambda: regex.sub('(?P<a>x)', '\\g<-1>', 'xx'))