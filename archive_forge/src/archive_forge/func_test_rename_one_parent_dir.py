import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_parent_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/c'])
    tree.add(['a', 'b', 'b/c'])
    tree.commit('initial')
    c_contents = tree.get_file_text('b/c')
    tree.rename_one('b/c', 'd')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('d', 'b/c')])
    self.assertPathDoesNotExist('b/c')
    self.assertFileEqual(c_contents, 'd')