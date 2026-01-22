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
def test_commit_exclude_subtree_of_selected(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'a/c'])
    tree.smart_add(['.'])
    tree.commit('test', specific_files=['a', 'a/c'], exclude=['a/b'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    changes = list(tree.iter_changes(tree.basis_tree()))
    self.assertEqual(1, len(changes), changes)
    self.assertEqual((None, 'a/b'), changes[0].path)