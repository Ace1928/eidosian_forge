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
def test_commit_deleted_subtree_with_removed(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/c', 'd'])
    wt.add(['a', 'b', 'b/c'])
    wt.commit('first')
    wt.remove('b/c')
    this_dir = wt.controldir.root_transport
    this_dir.delete_tree('b')
    with wt.lock_write():
        wt.commit('commit deleted rename')
        self.assertTrue(wt.is_versioned('a'))
        self.assertTrue(wt.has_filename('a'))
        self.assertFalse(wt.has_filename('b'))
        self.assertFalse(wt.has_filename('b/c'))