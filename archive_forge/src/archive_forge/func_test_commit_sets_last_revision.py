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
def test_commit_sets_last_revision(self):
    tree = self.make_branch_and_tree('tree')
    if tree.branch.repository._format.supports_setting_revision_ids:
        committed_id = tree.commit('foo', rev_id=b'foo')
        self.assertEqual(b'foo', committed_id)
    else:
        committed_id = tree.commit('foo')
    self.assertEqual([committed_id], tree.get_parent_ids())