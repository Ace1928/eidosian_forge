import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_to_branch_no_working_tree(self):
    branch1_tree = self.example_branch2()
    dir1 = branch1_tree.controldir
    dir1.destroy_workingtree()
    self.assertFalse(dir1.has_workingtree())
    output = self.run_bzr('diff -r 1.. branch1', retcode=1)
    self.assertContainsRe(output[0], '\n\\-original line\n\\+repo line\n')