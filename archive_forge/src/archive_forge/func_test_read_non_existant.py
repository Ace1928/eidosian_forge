import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_read_non_existant(self):
    manager = self.get_manager()
    e = self.assertRaises(shelf.NoSuchShelfId, manager.read_shelf, 1)
    self.assertEqual('No changes are shelved with id "1".', str(e))