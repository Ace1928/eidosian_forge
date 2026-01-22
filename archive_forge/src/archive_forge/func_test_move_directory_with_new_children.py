import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory_with_new_children(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/c', 'b/'])
    tree.add(['a', 'b', 'a/c'])
    tree.commit('initial')
    self.build_tree(['a/b', 'a/d'])
    tree.add(['a/b', 'a/d'])
    self.assertEqual([('a', 'b/a')], tree.move(['a'], 'b'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/a/', 'a/'), ('b/a/b', None), ('b/a/c', 'a/c'), ('b/a/d', None)])
    tree._validate()