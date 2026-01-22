import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_arbitrary(self):
    target = self.make_branch_and_tree('target')
    target.commit('empty')
    branch_a = target.controldir.sprout('branch_a').open_workingtree()
    self.build_tree(['branch_a/file1'])
    branch_a.add('file1')
    branch_a.commit('added file1', rev_id=b'rev2a')
    branch_b = target.controldir.sprout('branch_b').open_workingtree()
    self.build_tree(['branch_b/file2'])
    branch_b.add('file2')
    branch_b.commit('added file2', rev_id=b'rev2b')
    branch_b.merge_from_branch(branch_a.branch)
    self.assertPathExists('branch_b/file1')
    branch_b.commit('merged branch_a', rev_id=b'rev3b')
    self.run_bzr('merge -d target -r revid:rev2a branch_a')
    self.assertPathExists('target/file1')
    self.assertPathDoesNotExist('target/file2')
    target.revert()
    self.run_bzr('merge -d target -r revid:rev2a branch_b')
    self.assertPathExists('target/file1')
    self.assertPathDoesNotExist('target/file2')