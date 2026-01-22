import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_refcount(self):
    f = 'fo' + 'oo'
    num_refs = sys.getrefcount(f) - 1
    k = self.module.StaticTuple(f)
    self.assertRefcount(num_refs + 1, f)
    b = k[0]
    self.assertRefcount(num_refs + 2, f)
    b = k[0]
    self.assertRefcount(num_refs + 2, f)
    c = k[0]
    self.assertRefcount(num_refs + 3, f)
    del b, c
    self.assertRefcount(num_refs + 1, f)
    del k
    self.assertRefcount(num_refs, f)