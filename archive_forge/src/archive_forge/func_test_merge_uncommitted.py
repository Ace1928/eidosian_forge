import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_uncommitted(self):
    """Check that merge --uncommitted behaves properly"""
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/file_1', 'a/file_2'])
    tree_a.add(['file_1', 'file_2'])
    tree_a.commit('commit 1')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    self.assertPathExists('b/file_1')
    tree_a.rename_one('file_1', 'file_i')
    tree_a.commit('commit 2')
    tree_a.rename_one('file_2', 'file_ii')
    self.run_bzr('merge a --uncommitted -d b')
    self.assertPathExists('b/file_1')
    self.assertPathExists('b/file_ii')
    tree_b.revert()
    self.run_bzr_error(('Cannot use --uncommitted and --revision',), 'merge /a --uncommitted -r1 -d b')