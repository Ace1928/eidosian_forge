import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_referents(self):
    self.requireFeature(features.meliae)
    from meliae import scanner
    strs = ['foo', 'bar', 'baz', 'bing']
    k = self.module.StaticTuple(*strs)
    if self.module is _static_tuple_py:
        refs = strs + [self.module.StaticTuple]
    else:
        refs = strs

    def key(k):
        if isinstance(k, type):
            return (0, k)
        if isinstance(k, str):
            return (1, k)
        raise TypeError(k)
    self.assertEqual(sorted(refs, key=key), sorted(scanner.get_referents(k), key=key))