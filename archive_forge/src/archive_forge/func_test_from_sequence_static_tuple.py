import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_from_sequence_static_tuple(self):
    st = self.module.StaticTuple('foo', 'bar')
    st2 = self.module.StaticTuple.from_sequence(st)
    self.assertIs(st, st2)