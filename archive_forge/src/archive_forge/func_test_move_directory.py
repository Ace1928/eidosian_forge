import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_directory(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c/', 'a/c/d', 'e/'])
    tree.add(['a', 'a/b', 'a/c', 'a/c/d', 'e'])
    tree.commit('initial')
    self.assertEqual([('a', 'e/a')], tree.move(['a'], 'e'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('e/', 'e/'), ('e/a/', 'a/'), ('e/a/b', 'a/b'), ('e/a/c/', 'a/c/'), ('e/a/c/d', 'a/c/d')])
    tree._validate()