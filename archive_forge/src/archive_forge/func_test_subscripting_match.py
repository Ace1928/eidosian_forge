from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_subscripting_match(self):
    m = regex.match('(?<a>\\w)', 'xy')
    if not m:
        self.fail('Failed: expected match but returned None')
    elif not m or m[0] != m.group(0) or m[1] != m.group(1):
        self.fail('Failed')
    if not m:
        self.fail('Failed: expected match but returned None')
    elif m[:] != ('x', 'x'):
        self.fail('Failed: expected "(\'x\', \'x\')" but got {} instead'.format(ascii(m[:])))