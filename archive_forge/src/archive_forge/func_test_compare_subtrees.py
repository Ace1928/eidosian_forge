import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_compare_subtrees(self):
    tree1 = self.make_branch_and_tree('1')
    if not tree1.supports_tree_reference():
        return
    tree1.set_root_id(b'root-id')
    subtree1 = self.make_branch_and_tree('1/sub')
    subtree1.set_root_id(b'subtree-id')
    tree1.add_reference(subtree1)
    tree2 = self.make_to_branch_and_tree('2')
    if not tree2.supports_tree_reference():
        return
    tree2.set_root_id(b'root-id')
    subtree2 = self.make_to_branch_and_tree('2/sub')
    subtree2.set_root_id(b'subtree-id')
    tree2.add_reference(subtree2)
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual([], list(tree2.iter_changes(tree1)))
    subtree1.commit('commit', rev_id=b'commit-a')
    self.assertThat(tree2.iter_changes(tree1, include_unchanged=True), MatchesTreeChanges(tree1, tree2, [TreeChange(('', ''), False, (True, True), ('', ''), ('directory', 'directory'), (False, False)), TreeChange(('sub', 'sub'), False, (True, True), ('sub', 'sub'), ('tree-reference', 'tree-reference'), (False, False))]))