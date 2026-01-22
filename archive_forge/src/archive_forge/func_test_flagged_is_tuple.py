import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_flagged_is_tuple(self):
    debug.debug_flags.add('static_tuple')
    t = ('foo',)
    self.assertRaises(TypeError, static_tuple.expect_static_tuple, t)