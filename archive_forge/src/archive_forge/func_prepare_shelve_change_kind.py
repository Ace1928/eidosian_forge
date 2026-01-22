import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_shelve_change_kind(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/foo', b'bar')])
    tree.add('foo', ids=b'foo-id')
    tree.commit('Added file and directory')
    os.unlink('tree/foo')
    os.mkdir('tree/foo')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('change kind', b'foo-id', 'file', 'directory', 'foo')], sorted(list(creator.iter_shelvable())))
    return creator