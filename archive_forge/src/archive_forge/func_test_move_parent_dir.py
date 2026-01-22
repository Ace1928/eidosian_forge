import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_parent_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/c'])
    tree.add(['a', 'b', 'b/c'])
    tree.commit('initial')
    c_contents = tree.get_file_text('b/c')
    self.assertEqual([('b/c', 'c')], tree.move(['b/c'], ''))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('c', 'b/c')])
    self.assertPathDoesNotExist('b/c')
    self.assertFileEqual(c_contents, 'c')
    tree._validate()