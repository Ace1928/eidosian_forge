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
def test_commit_move_new(self):
    wt = self.make_branch_and_tree('first')
    wt.commit('first')
    wt2 = wt.controldir.sprout('second').open_workingtree()
    self.build_tree(['second/name1'])
    wt2.add('name1')
    wt2.commit('second')
    wt.merge_from_branch(wt2.branch)
    wt.rename_one('name1', 'name2')
    wt.commit('third')
    self.assertFalse(wt.is_versioned('name1'))