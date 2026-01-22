import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_from_sequence_tuple(self):
    st = self.module.StaticTuple.from_sequence(('foo', 'bar'))
    self.assertIsInstance(st, self.module.StaticTuple)
    self.assertEqual(('foo', 'bar'), st)