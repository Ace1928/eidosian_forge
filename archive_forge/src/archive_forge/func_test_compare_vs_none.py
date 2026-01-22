import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_vs_none(self):
    k1 = self.module.StaticTuple('baz', 'bing')
    self.assertCompareDifferent(None, k1, mismatched_types=True)