import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_files_into_unversioned_subdir(self):
    """Test brz mv original_file to unversioned_directory.

        Tests if an attempt to move existing versioned file
        into an unversioned directory will fail.
        Setup: a1, a2 are in the working tree, sub is not.
        User does: mv a1 sub/.; brz mv a1 a2 sub
        """
    self.build_tree(['a1', 'a2', 'sub/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a1', 'a2'])
    osutils.rename('a1', 'sub/a1')
    self.run_bzr_error(['^brz: ERROR: Could not move to sub. sub is not versioned\\.$'], 'mv a1 a2 sub')
    self.assertPathDoesNotExist('a1')
    self.assertPathExists('sub/a1')
    self.assertPathExists('a2')
    self.assertPathDoesNotExist('sub/a2')