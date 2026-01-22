import os
from io import BytesIO
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_tree import TestCaseWithTree
from ... import revision as _mod_revision
from ... import tests, trace
from ...diff import show_diff_trees
from ...merge import Merge3Merger, Merger
from ...transform import ROOT_PARENT, resolve_conflicts
from ...tree import TreeChange, find_previous_path
from ..features import SymlinkFeature, UnicodeFilenameFeature
def test_path2id_moved(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/old_parent/', 'tree/old_parent/child'])
    tree.add(['old_parent', 'old_parent/child'])
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    new_parent = preview.new_directory('new_parent', preview.root, b'new_parent-id')
    preview.adjust_path('child', new_parent, preview.trans_id_tree_path('old_parent/child'))
    preview_tree = preview.get_preview_tree()
    self.assertFalse(preview_tree.is_versioned('old_parent/child'))
    self.assertEqual('new_parent/child', find_previous_path(tree, preview_tree, 'old_parent/child'))
    if self.workingtree_format.supports_setting_file_ids:
        self.assertEqual(tree.path2id('old_parent/child'), preview_tree.path2id('new_parent/child'))