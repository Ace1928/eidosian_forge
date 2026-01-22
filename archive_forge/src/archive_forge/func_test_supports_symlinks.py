from breezy import osutils, tests
from breezy.git.branch import GitBranch
from breezy.mutabletree import MutableTree
from breezy.tests import TestSkipped, features, per_tree
from breezy.transform import PreviewTree
def test_supports_symlinks(self):
    self.tree = self.make_branch_and_tree('.')
    self.assertIn(self.tree.supports_symlinks(), [True, False])