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
def test_iter_entries_by_dir_new(self):
    tree = self.make_branch_and_tree('tree')
    tt = tree.transform()
    tt.new_file('new', tt.root, [b'contents'], b'new-id')
    self.assertMatchingIterEntries(tt)