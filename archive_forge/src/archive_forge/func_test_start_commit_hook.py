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
def test_start_commit_hook(self):
    """Make sure a start commit hook can modify the tree that is
        committed."""

    def start_commit_hook_adds_file(tree):
        with open(tree.abspath('newfile'), 'w') as f:
            f.write('data')
        tree.add(['newfile'])

    def restoreDefaults():
        mutabletree.MutableTree.hooks['start_commit'] = []
    self.addCleanup(restoreDefaults)
    tree = self.make_branch_and_tree('.')
    mutabletree.MutableTree.hooks.install_named_hook('start_commit', start_commit_hook_adds_file, None)
    revid = tree.commit('first post')
    committed_tree = tree.basis_tree()
    self.assertTrue(committed_tree.has_filename('newfile'))