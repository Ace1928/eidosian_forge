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
def test_commit_progress_shows_post_hook_names(self):
    tree = self.make_branch_and_tree('.')
    factory = ProgressRecordingUIFactory()
    ui.ui_factory = factory

    def a_hook(_, _2, _3, _4, _5, _6):
        pass
    branch.Branch.hooks.install_named_hook('post_commit', a_hook, 'hook name')
    tree.commit('first post')
    self.assertEqual([('update', 1, 5, 'Collecting changes [0] - Stage'), ('update', 1, 5, 'Collecting changes [1] - Stage'), ('update', 2, 5, 'Saving data locally - Stage'), ('update', 3, 5, 'Running pre_commit hooks - Stage'), ('update', 4, 5, 'Updating the working tree - Stage'), ('update', 5, 5, 'Running post_commit hooks - Stage'), ('update', 5, 5, 'Running post_commit hooks [hook name] - Stage')], factory._calls)