import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_from_sequence_not_sequence(self):
    self.assertRaises(TypeError, self.module.StaticTuple.from_sequence, object())
    self.assertRaises(TypeError, self.module.StaticTuple.from_sequence, 10)