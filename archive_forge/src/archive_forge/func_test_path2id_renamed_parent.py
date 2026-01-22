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
def test_path2id_renamed_parent(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/old_name/', 'tree/old_name/child'])
    tree.add(['old_name', 'old_name/child'])
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.adjust_path('new_name', preview.root, preview.trans_id_tree_path('old_name'))
    preview_tree = preview.get_preview_tree()
    self.assertFalse(preview_tree.is_versioned('old_name/child'))
    self.assertEqual('new_name/child', find_previous_path(tree, preview_tree, 'old_name/child'))
    if tree.supports_setting_file_ids():
        self.assertEqual(tree.path2id('old_name/child'), preview_tree.path2id('new_name/child'))