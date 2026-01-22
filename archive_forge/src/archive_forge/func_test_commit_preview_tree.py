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
def test_commit_preview_tree(self):
    tree = self.make_branch_and_tree('tree')
    rev_id = tree.commit('rev1')
    tree.branch.lock_write()
    self.addCleanup(tree.branch.unlock)
    tt = tree.preview_transform()
    tt.new_file('file', tt.root, [b'contents'], b'file_id')
    self.addCleanup(tt.finalize)
    preview = tt.get_preview_tree()
    preview.set_parent_ids([rev_id])
    builder = tree.branch.get_commit_builder([rev_id])
    list(builder.record_iter_changes(preview, rev_id, tt.iter_changes()))
    builder.finish_inventory()
    rev2_id = builder.commit('rev2')
    rev2_tree = tree.branch.repository.revision_tree(rev2_id)
    self.assertEqual(b'contents', rev2_tree.get_file_text('file'))