import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_static_tuple_thunk(self):
    if self.module is _static_tuple_py:
        if compiled_static_tuple_feature.available():
            return
    self.assertIs(static_tuple.StaticTuple, self.module.StaticTuple)