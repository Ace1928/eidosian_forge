import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_fail_consistent(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/a', 'c'])
    tree.add(['a', 'b', 'c'])
    tree.commit('initial')
    self.assertRaises(errors.RenameFailedFilesExist, tree.rename_one, 'a', 'b/a')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('c', 'c')])