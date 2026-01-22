import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_last_shelf(self):
    manager = self.get_manager()
    self.assertIs(None, manager.last_shelf())
    shelf_id, shelf_file = manager.new_shelf()
    shelf_file.close()
    self.assertEqual(1, manager.last_shelf())