import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert_dirname(self):
    """Test that revert DIRECTORY does what's expected"""
    self._prepare_rename_mod_tree()
    self.run_bzr('revert a')
    self.assertPathExists('a/b')
    self.assertPathExists('a/d')
    self.assertPathDoesNotExist('a/g')
    self.expectFailure('j is in the delta revert applies because j was renamed too', self.assertPathExists, 'j')
    self.assertPathExists('h')
    self.run_bzr('revert f')
    self.assertPathDoesNotExist('j')
    self.assertPathDoesNotExist('h')
    self.assertPathExists('a/d/e')