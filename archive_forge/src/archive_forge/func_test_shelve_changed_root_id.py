import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_changed_root_id(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.set_root_id(b'first-root-id')
    tree.add(['foo'], ids=[b'foo-id'])
    tree.commit('foo')
    tree.set_root_id(b'second-root-id')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.expectFailure("shelf doesn't support shelving root changes yet", self.assertEqual, [('delete file', b'first-root-id', 'directory', ''), ('add file', b'second-root-id', 'directory', ''), ('rename', b'foo-id', 'foo', 'foo')], list(creator.iter_shelvable()))
    self.assertEqual([('delete file', b'first-root-id', 'directory', ''), ('add file', b'second-root-id', 'directory', ''), ('rename', b'foo-id', 'foo', 'foo')], list(creator.iter_shelvable()))