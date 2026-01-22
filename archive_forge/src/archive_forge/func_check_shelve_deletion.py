import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def check_shelve_deletion(self, tree):
    self.assertEqual(tree.id2path(b'foo-id'), 'foo')
    self.assertEqual(tree.id2path(b'bar-id'), 'foo/bar')
    self.assertFileEqual(b'baz', 'tree/foo/bar')