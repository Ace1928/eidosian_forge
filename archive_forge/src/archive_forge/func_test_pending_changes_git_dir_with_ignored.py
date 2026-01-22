import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_pending_changes_git_dir_with_ignored(self):
    tree = self.make_test_tree(format='git')
    self.build_tree_contents([('debian/upstream/',), ('debian/upstream/blah', ''), ('.gitignore', 'blah\n')])
    tree.add('.gitignore')
    tree.commit('add gitignore')
    with tree.lock_write():
        check_clean_tree(tree)