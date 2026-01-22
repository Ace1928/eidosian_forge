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
def test_commit_exclude_exclude_changed_is_pointless(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.smart_add(['.'])
    tree.commit('setup test')
    self.build_tree_contents([('a', b'new contents for "a"\n')])
    self.assertRaises(PointlessCommit, tree.commit, 'test', exclude=['a'], allow_pointless=False)