import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_defaults_to_reprocess(self):
    tree, other = self.create_conflicting_branches()
    self.run_bzr('merge ../other', working_dir='tree', retcode=1)
    self.assertEqualDiff(b'a\nB\n<<<<<<< TREE\nC\n=======\nD\n>>>>>>> MERGE-SOURCE\n', tree.get_file_text('fname'))