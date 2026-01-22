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
def test_is_executable2(self):
    tree = self.make_branch_and_tree('tree')
    preview = tree.preview_transform()
    self.addCleanup(preview.finalize)
    preview.new_file('foo', preview.root, [b'bar'], b'baz-id')
    preview_tree = preview.get_preview_tree()
    self.assertEqual(False, preview_tree.is_executable('tree/foo'))