import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_specific_with_rename_under_dir_under_new_dir_reports_new_dir(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_abc_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content_7(tree2)
    tree2.rename_one('a', 'd/e/a')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    root_id = tree1.path2id('')
    self.assertEqualIterChanges([self.renamed(tree1, tree2, 'b', 'd/e', False), self.added(tree2, 'd'), self.renamed(tree1, tree2, 'a', 'd/e/a', False)], self.do_iter_changes(tree1, tree2, specific_files=['d/e/a']))