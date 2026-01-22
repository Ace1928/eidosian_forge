import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_deletion(self):
    creator, tree = self.prepare_shelve_deletion()
    creator.shelve_change(('delete file', b'foo-id', 'directory', 'foo'))
    creator.shelve_change(('delete file', b'bar-id', 'file', 'foo/bar'))
    creator.transform()
    self.check_shelve_deletion(tree)