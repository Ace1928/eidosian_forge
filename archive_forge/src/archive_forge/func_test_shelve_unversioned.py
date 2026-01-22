import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_unversioned(self):
    tree = self.make_branch_and_tree('tree')
    with tree.lock_tree_write():
        self.assertRaises(errors.PathsNotVersionedError, shelf.ShelfCreator, tree, tree.basis_tree(), ['foo'])
    wt = workingtree.WorkingTree.open('tree')
    wt.lock_tree_write()
    wt.unlock()
    with tree.lock_tree_write():
        self.assertRaises(errors.PathsNotVersionedError, shelf.ShelfCreator, tree, tree.basis_tree(), ['foo'])