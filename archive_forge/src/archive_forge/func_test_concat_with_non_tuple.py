import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_concat_with_non_tuple(self):
    st1 = self.module.StaticTuple('foo')
    self.assertRaises(TypeError, lambda: st1 + 10)