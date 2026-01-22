import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_directory(self):
    """Use `brz mv a b` to mark a directory as renamed.

        https://bugs.launchpad.net/bzr/+bug/107967/
        """
    self.build_tree(['a/', 'c/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a', 'c'])
    osutils.rename('a', 'b')
    osutils.rename('c', 'd')
    self.run_bzr('mv a b')
    self.assertPathDoesNotExist('a')
    self.assertNotInWorkingTree('a')
    self.assertPathExists('b')
    self.assertInWorkingTree('b')
    self.run_bzr('mv --after c d')
    self.assertPathDoesNotExist('c')
    self.assertNotInWorkingTree('c')
    self.assertPathExists('d')
    self.assertInWorkingTree('d')