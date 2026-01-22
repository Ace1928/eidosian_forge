import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_versioned_symlinks_specific_files(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree1, tree2 = self.make_trees_with_symlinks()
    root_id = tree1.path2id('')
    expected = [self.added(tree2, 'added'), self.changed_content(tree2, 'changed'), self.kind_changed(tree1, tree2, 'fromdir', 'fromdir'), self.kind_changed(tree1, tree2, 'fromfile', 'fromfile'), self.deleted(tree1, 'removed'), self.kind_changed(tree1, tree2, 'todir', 'todir'), self.kind_changed(tree1, tree2, 'tofile', 'tofile')]
    expected = self.sorted(expected)
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['added', 'changed', 'fromdir', 'fromfile', 'removed', 'unchanged', 'todir', 'tofile']))
    self.check_has_changes(True, tree1, tree2)