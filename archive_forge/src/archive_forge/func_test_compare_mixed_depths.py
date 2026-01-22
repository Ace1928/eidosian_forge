import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_mixed_depths(self):
    stuple = self.module.StaticTuple
    k1 = stuple(stuple('a'), stuple('b'))
    k2 = stuple(stuple(stuple('c'), stuple('d')), stuple('b'))
    self.assertCompareNoRelation(k1, k2, mismatched_types=True)