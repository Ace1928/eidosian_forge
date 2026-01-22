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
def test_record_two_ghosts(self):
    """The working tree should preserve all the parents during commit."""
    wt = self.make_branch_and_tree('.')
    if not wt.branch.repository._format.supports_ghosts:
        raise tests.TestNotApplicable('format does not support ghosts')
    wt.set_parent_ids([b'foo@azkhazan-123123-abcabc', b'wibble@fofof--20050401--1928390812'], allow_leftmost_as_ghost=True)
    rev_id = wt.commit('commit from ghost base with one merge')
    rev = wt.branch.repository.get_revision(rev_id)
    self.assertEqual([b'foo@azkhazan-123123-abcabc', b'wibble@fofof--20050401--1928390812'], rev.parent_ids)