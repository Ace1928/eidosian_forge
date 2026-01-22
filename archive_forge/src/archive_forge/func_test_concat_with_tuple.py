import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_concat_with_tuple(self):
    st1 = self.module.StaticTuple('foo')
    t2 = ('bar',)
    st3 = self.module.StaticTuple('foo', 'bar')
    st4 = self.module.StaticTuple('bar', 'foo')
    st5 = st1 + t2
    st6 = t2 + st1
    self.assertEqual(st3, st5)
    self.assertIsInstance(st5, self.module.StaticTuple)
    self.assertEqual(st4, st6)
    if self.module is _static_tuple_py:
        self.assertIsInstance(st6, tuple)
    else:
        self.assertIsInstance(st6, self.module.StaticTuple)