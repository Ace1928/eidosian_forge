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
def test_deleted_content_summary(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/path/'])
    tree.add('path')
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.delete_contents(preview.trans_id_tree_path('path'))
    summary = preview.get_preview_tree().path_content_summary('path')
    self.assertEqual(('missing', None, None, None), summary)