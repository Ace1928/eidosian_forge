import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_shelve_rename(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add(['foo'], ids=[b'foo-id'])
    tree.commit('foo')
    tree.rename_one('foo', 'bar')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('rename', b'foo-id', 'foo', 'bar')], list(creator.iter_shelvable()))
    return creator