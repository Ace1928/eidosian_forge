import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_hold_bool(self):
    k1 = self.module.StaticTuple(True)
    k2 = self.module.StaticTuple(False)