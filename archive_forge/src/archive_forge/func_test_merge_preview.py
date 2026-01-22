import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_preview(self):
    this_tree = self.make_branch_and_tree('this')
    this_tree.commit('rev1')
    other_tree = this_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/file', b'new line')])
    other_tree.add('file')
    other_tree.commit('rev2a')
    this_tree.commit('rev2b')
    out, err = self.run_bzr(['merge', '-d', 'this', 'other', '--preview'])
    self.assertContainsRe(out, '\\+new line')
    self.assertNotContainsRe(err, '\\+N  file\n')
    this_tree.lock_read()
    self.addCleanup(this_tree.unlock)
    self.assertEqual([], list(this_tree.iter_changes(this_tree.basis_tree())))