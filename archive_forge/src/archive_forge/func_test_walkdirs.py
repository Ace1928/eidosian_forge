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
def test_walkdirs(self):
    preview = self.get_empty_preview()
    preview.new_directory('', ROOT_PARENT, b'tree-root')
    preview.fixup_new_roots()
    preview_tree = preview.get_preview_tree()
    preview.new_file('a', preview.root, [b'contents'], b'a-id')
    expected = [('', [('a', 'a', 'file', None, 'file')])]
    self.assertEqual(expected, list(preview_tree.walkdirs()))