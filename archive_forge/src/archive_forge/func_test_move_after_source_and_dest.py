import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_after_source_and_dest(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/a'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    with open('a') as a_file:
        a_text = a_file.read()
    with open('b/a') as ba_file:
        ba_text = ba_file.read()
    self.assertTreeLayout(['', 'a', 'b/'], tree)
    self.assertRaises(errors.RenameFailedFilesExist, tree.move, ['a'], 'b', after=False)
    self.assertTreeLayout(['', 'a', 'b/'], tree)
    self.assertFileEqual(a_text, 'a')
    self.assertFileEqual(ba_text, 'b/a')
    self.assertEqual([('a', 'b/a')], tree.move(['a'], 'b', after=True))
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/a', 'a')])
    self.assertFileEqual(a_text, 'a')
    self.assertFileEqual(ba_text, 'b/a')
    tree._validate()