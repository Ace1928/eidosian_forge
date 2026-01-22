import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_pull(self):
    self.pullable_branch()
    out, err = self.run_bzr('merge --pull ../b', working_dir='a')
    self.assertContainsRe(out, 'Now on revision 2\\.')
    tree_a = workingtree.WorkingTree.open('a')
    self.assertEqual([self.id2], tree_a.get_parent_ids())