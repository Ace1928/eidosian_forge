import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_disk_in_subtrees_skipped(self):
    """subtrees are considered not-in-the-current-tree.

        This test tests the trivial case, where the basis has no paths in the
        current trees subtree.
        """
    tree1 = self.make_branch_and_tree('1')
    tree1.set_root_id(b'root-id')
    tree2 = self.make_to_branch_and_tree('2')
    if not tree2.supports_tree_reference():
        return
    tree2.set_root_id(b'root-id')
    subtree2 = self.make_to_branch_and_tree('2/sub')
    subtree2.set_root_id(b'subtree-id')
    tree2.add_reference(subtree2)
    self.build_tree(['2/sub/file'])
    subtree2.add(['file'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual([self.added(tree2, 'sub')], self.do_iter_changes(tree1, tree2, want_unversioned=True))
    self.assertEqual([self.added(tree2, 'sub')], self.do_iter_changes(tree1, tree2, specific_files=['sub'], want_unversioned=True))