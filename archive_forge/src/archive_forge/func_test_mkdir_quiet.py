import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_mkdir_quiet(self):
    """'brz mkdir --quiet' should not print a status message"""
    self.make_branch_and_tree('.')
    out, err = self.run_bzr(['mkdir', '--quiet', 'foo'])
    self.assertEqual('', err)
    self.assertEqual('', out)