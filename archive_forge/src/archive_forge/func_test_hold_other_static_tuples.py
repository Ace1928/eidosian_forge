import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_hold_other_static_tuples(self):
    k = self.module.StaticTuple('foo', 'bar')
    k2 = self.module.StaticTuple(k, k)
    self.assertEqual(2, len(k2))
    self.assertIs(k, k2[0])
    self.assertIs(k, k2[1])