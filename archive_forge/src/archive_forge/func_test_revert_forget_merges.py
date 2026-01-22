import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert_forget_merges(self):
    tree = self.make_branch_and_tree('.')
    self.run_bzr(['revert', '--forget-merges'])
    self.build_tree(['file'])
    first_rev_id = tree.commit('initial commit')
    self.build_tree_contents([('file', b'new content')])
    existing_parents = tree.get_parent_ids()
    self.assertEqual([first_rev_id], existing_parents)
    merged_parents = existing_parents + [b'merged-in-rev']
    tree.set_parent_ids(merged_parents)
    self.assertEqual(merged_parents, tree.get_parent_ids())
    self.run_bzr(['revert', '--forget-merges'])
    self.assertEqual([first_rev_id], tree.get_parent_ids())
    self.assertFileEqual(b'new content', 'file')
    self.run_bzr(['revert', '--forget-merges', tree.abspath('.')])