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
def test_post_commit_hook(self):
    """Make sure a post_commit hook is called after a commit."""

    def post_commit_hook_test_params(params):
        self.assertTrue(isinstance(params, mutabletree.PostCommitHookParams))
        self.assertTrue(isinstance(params.mutable_tree, mutabletree.MutableTree))
        with open(tree.abspath('newfile'), 'w') as f:
            f.write('data')
        params.mutable_tree.add(['newfile'])
    tree = self.make_branch_and_tree('.')
    mutabletree.MutableTree.hooks.install_named_hook('post_commit', post_commit_hook_test_params, None)
    self.assertFalse(tree.has_filename('newfile'))
    revid = tree.commit('first post')
    self.assertTrue(tree.has_filename('newfile'))
    committed_tree = tree.basis_tree()
    self.assertFalse(committed_tree.has_filename('newfile'))