import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory_with_swapped_children(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c', 'a/d', 'e/'])
    tree.add(['a', 'a/b', 'a/c', 'a/d', 'e'])
    tree.commit('initial')
    tree.rename_one('a/b', 'a/bb')
    tree.rename_one('a/d', 'a/b')
    tree.rename_one('a/bb', 'a/d')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('e/', 'e/'), ('a/b', 'a/d'), ('a/c', 'a/c'), ('a/d', 'a/b')])
    self.assertEqual([('a', 'e/a')], tree.move(['a'], 'e'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('e/', 'e/'), ('e/a/', 'a/'), ('e/a/b', 'a/d'), ('e/a/c', 'a/c'), ('e/a/d', 'a/b')])
    tree._validate()