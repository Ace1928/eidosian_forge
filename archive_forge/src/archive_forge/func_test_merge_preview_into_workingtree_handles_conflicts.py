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
def test_merge_preview_into_workingtree_handles_conflicts(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/foo', b'bar')])
    tree.add('foo')
    tree.commit('foo')
    tt = tree.preview_transform()
    self.addCleanup(tt.finalize)
    trans_id = tt.trans_id_tree_path('foo')
    tt.delete_contents(trans_id)
    tt.create_file([b'baz'], trans_id)
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    self.build_tree_contents([('tree2/foo', b'qux')])
    merger = Merger.from_uncommitted(tree2, tt.get_preview_tree(), tree.basis_tree())
    merger.merge_type = Merge3Merger
    merger.do_merge()