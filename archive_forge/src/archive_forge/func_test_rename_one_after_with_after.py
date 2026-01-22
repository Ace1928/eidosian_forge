import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_after_with_after(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    os.rename('a', 'b/foo')
    if tree.has_versioned_directories():
        self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/')])
    else:
        self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a')])
    tree.rename_one('a', 'b/foo', after=True)
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/foo', 'a')])