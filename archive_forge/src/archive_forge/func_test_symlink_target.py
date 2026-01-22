from breezy import osutils, tests
from breezy.git.branch import GitBranch
from breezy.mutabletree import MutableTree
from breezy.tests import TestSkipped, features, per_tree
from breezy.transform import PreviewTree
def test_symlink_target(self):
    if isinstance(self.tree, (MutableTree, PreviewTree)):
        raise TestSkipped('symlinks not accurately represented in working trees and preview trees')
    entry = get_entry(self.tree, 'symlink')
    self.assertEqual(entry.symlink_target, 'link-target')