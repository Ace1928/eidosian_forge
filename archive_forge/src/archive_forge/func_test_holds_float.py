import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_holds_float(self):
    k1 = self.module.StaticTuple(1.2)

    class subfloat(float):
        pass
    self.assertRaises(TypeError, self.module.StaticTuple, subfloat(1.5))