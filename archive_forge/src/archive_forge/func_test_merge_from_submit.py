import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_from_submit(self):
    tree_a = self.make_branch_and_tree('a')
    tree_a.commit('test')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    tree_c = tree_a.controldir.sprout('c').open_workingtree()
    out, err = self.run_bzr(['merge', '-d', 'c'])
    self.assertContainsRe(err, 'Merging from remembered parent location .*a\\/')
    with tree_c.branch.lock_write():
        tree_c.branch.set_submit_branch(tree_b.controldir.root_transport.base)
    out, err = self.run_bzr(['merge', '-d', 'c'])
    self.assertContainsRe(err, 'Merging from remembered submit location .*b\\/')