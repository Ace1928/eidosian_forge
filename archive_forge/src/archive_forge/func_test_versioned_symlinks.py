import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_versioned_symlinks(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree1, tree2 = self.make_trees_with_symlinks()
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    root_id = tree1.path2id('')
    expected = [self.unchanged(tree1, ''), self.added(tree2, 'added'), self.changed_content(tree2, 'changed'), self.kind_changed(tree1, tree2, 'fromdir', 'fromdir'), self.kind_changed(tree1, tree2, 'fromfile', 'fromfile'), self.deleted(tree1, 'removed'), self.unchanged(tree2, 'unchanged'), self.unversioned(tree2, 'unknown'), self.kind_changed(tree1, tree2, 'todir', 'todir'), self.kind_changed(tree1, tree2, 'tofile', 'tofile')]
    expected = self.sorted(expected)
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, include_unchanged=True, want_unversioned=True))
    self.check_has_changes(True, tree1, tree2)