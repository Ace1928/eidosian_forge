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
def test_iter_entries_by_dir_moved(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/moved', 'tree/new_parent/'])
    tree.add(['moved', 'new_parent'])
    tt = tree.transform()
    tt.adjust_path('moved', tt.trans_id_tree_path('new_parent'), tt.trans_id_tree_path('moved'))
    self.assertMatchingIterEntries(tt)