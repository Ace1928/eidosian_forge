import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_over_deleted(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'b'])
    tree.add(['a', 'a/b', 'b'])
    tree.commit('initial')
    tree.remove(['a/b'], keep_files=False)
    self.assertEqual([('b', 'a/b')], tree.move(['b'], 'a'))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('a/b', 'b')])
    tree._validate()