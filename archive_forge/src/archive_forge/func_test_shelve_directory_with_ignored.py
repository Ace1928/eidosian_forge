import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_directory_with_ignored(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.commit('Empty tree')
    self.build_tree_contents([('foo', b'a\n'), ('bar/',), ('bar/ignored', b'ign\n')])
    tree.add(['foo', 'bar'], ids=[b'foo-id', b'bar-id'])
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('add file', b'bar-id', 'directory', 'bar'), ('add file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
    ignores._set_user_ignores([])
    in_patterns = ['ignored']
    ignores.add_unique_user_ignores(in_patterns)
    creator.shelve_change(('add file', b'bar-id', 'directory', 'bar'))
    try:
        creator.transform()
        self.check_shelve_creation(creator, tree)
    except transform.MalformedTransform:
        raise KnownFailure('shelving directory with ignored file: see bug #611739')