import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_nonexistent_file(self):
    """It should not be possible to merge changes from a file which
        does not exist."""
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'bar\n')])
    tree_a.add(['file'])
    tree_a.commit('commit 1')
    self.run_bzr_error(('Path\\(s\\) do not exist: non/existing',), ['merge', 'non/existing'], working_dir='tree_a')