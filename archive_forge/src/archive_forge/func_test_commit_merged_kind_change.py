import os
from breezy import branch, conflicts, controldir, errors, mutabletree, osutils
from breezy import revision as _mod_revision
from breezy import tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.commit import CannotCommitSelectedFileMerge, PointlessCommit
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tests.testui import ProgressRecordingUIFactory
def test_commit_merged_kind_change(self):
    """Test merging a kind change.

        Test making a kind change in a working tree, and then merging that
        from another. When committed it should commit the new kind.
        """
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'])
    wt.commit('commit one')
    wt2 = wt.controldir.sprout('to').open_workingtree()
    os.remove('a')
    os.mkdir('a')
    wt.commit('changed kind')
    wt2.merge_from_branch(wt.branch)
    wt2.commit('merged kind change')