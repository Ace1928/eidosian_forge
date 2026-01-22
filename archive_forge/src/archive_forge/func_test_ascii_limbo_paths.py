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
def test_ascii_limbo_paths(self):
    self.requireFeature(UnicodeFilenameFeature)
    branch = self.make_branch('any')
    tree = branch.repository.revision_tree(_mod_revision.NULL_REVISION)
    tt = tree.preview_transform()
    self.addCleanup(tt.finalize)
    foo_id = tt.new_directory('', ROOT_PARENT)
    bar_id = tt.new_file('ሴbar', foo_id, [b'contents'])
    limbo_path = tt._limbo_name(bar_id)
    self.assertEqual(limbo_path, limbo_path)