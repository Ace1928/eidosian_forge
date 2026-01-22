import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_revision_info_tree_no_working_tree(self):
    b = self.make_branch('branch')
    out, err = self.run_bzr('revision-info --tree -d branch', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: No WorkingTree exists for "branch".\n', err)