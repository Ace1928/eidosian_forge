import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_weave_conflict_leaves_base_this_other_files(self):
    tree, other = self.create_conflicting_branches()
    self.run_bzr('merge ../other --weave', working_dir='tree', retcode=1)
    self.assertFileEqual(b'a\nb\nc\n', 'tree/fname.BASE')
    self.assertFileEqual(b'a\nB\nD\n', 'tree/fname.OTHER')
    self.assertFileEqual(b'a\nB\nC\n', 'tree/fname.THIS')