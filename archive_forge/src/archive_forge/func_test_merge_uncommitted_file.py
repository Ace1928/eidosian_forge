import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_uncommitted_file(self):
    """It should be possible to merge changes from a single file."""
    tree_a = self.make_branch_and_tree('tree_a')
    tree_a.commit('initial commit')
    tree_a.controldir.sprout('tree_b')
    self.build_tree(['tree_a/file1', 'tree_a/file2'])
    tree_a.add(['file1', 'file2'])
    self.run_bzr(['merge', '--uncommitted', '../tree_a/file1'], working_dir='tree_b')
    self.assertPathExists('tree_b/file1')
    self.assertPathDoesNotExist('tree_b/file2')