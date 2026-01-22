from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_subscripted_captures(self):
    self.assertEqual(regex.match('(?P<x>.)+', 'abc').expandf('{0} {0[0]} {0[-1]}'), 'abc abc abc')
    self.assertEqual(regex.match('(?P<x>.)+', 'abc').expandf('{1} {1[0]} {1[1]} {1[2]} {1[-1]} {1[-2]} {1[-3]}'), 'c a b c c b a')
    self.assertEqual(regex.match('(?P<x>.)+', 'abc').expandf('{x} {x[0]} {x[1]} {x[2]} {x[-1]} {x[-2]} {x[-3]}'), 'c a b c c b a')
    self.assertEqual(regex.subf('(?P<x>.)+', '{0} {0[0]} {0[-1]}', 'abc'), 'abc abc abc')
    self.assertEqual(regex.subf('(?P<x>.)+', '{1} {1[0]} {1[1]} {1[2]} {1[-1]} {1[-2]} {1[-3]}', 'abc'), 'c a b c c b a')
    self.assertEqual(regex.subf('(?P<x>.)+', '{x} {x[0]} {x[1]} {x[2]} {x[-1]} {x[-2]} {x[-3]}', 'abc'), 'c a b c c b a')