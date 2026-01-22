import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_get_metadata(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    shelf_manager = tree.get_shelf_manager()
    shelf_id = shelf_manager.shelve_changes(creator, 'foo')
    metadata = shelf_manager.get_metadata(shelf_id)
    self.assertEqual('foo', metadata[b'message'])
    self.assertEqual(b'null:', metadata[b'revision_id'])