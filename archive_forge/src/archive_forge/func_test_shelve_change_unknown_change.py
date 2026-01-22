import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_unknown_change(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    e = self.assertRaises(ValueError, creator.shelve_change, ('unknown',))
    self.assertEqual('Unknown change kind: "unknown"', str(e))