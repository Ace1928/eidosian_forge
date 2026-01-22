import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_different_types(self):
    k1 = self.module.StaticTuple('foo', 'bar')
    k2 = self.module.StaticTuple('foo', 1, None, 'µ', 1.2, 2 ** 65, True, k1)
    self.assertCompareNoRelation(k1, k2, mismatched_types=True)
    k3 = self.module.StaticTuple('foo')
    self.assertCompareDifferent(k3, k1)
    k4 = self.module.StaticTuple(None)
    self.assertCompareDifferent(k4, k1, mismatched_types=True)
    k5 = self.module.StaticTuple(1)
    self.assertCompareNoRelation(k1, k5, mismatched_types=True)