import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def check_tree_files(self, to_open, expected_tree, expect_paths):
    tree, relpaths = workingtree.WorkingTree.open_containing_paths(to_open)
    self.assertEndsWith(tree.basedir, expected_tree)
    self.assertEqual(expect_paths, relpaths)