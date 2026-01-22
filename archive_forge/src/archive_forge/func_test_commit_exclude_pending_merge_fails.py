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
def test_commit_exclude_pending_merge_fails(self):
    """Excludes are a form of partial commit."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    wt.add('foo')
    wt.commit('commit one')
    wt2 = wt.controldir.sprout('to').open_workingtree()
    wt2.commit('change_right')
    wt.merge_from_branch(wt2.branch)
    self.assertRaises(CannotCommitSelectedFileMerge, wt.commit, 'test', exclude=['foo'])