import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file_into_unversioned_subdir(self):
    """Test brz mv original_file to unversioned_directory/file.

        Tests if an attempt to move an existing versioned file
        into an unversioned directory will fail.
        Setup: a is in the working tree, sub/ is not.
        User does: mv a sub/a; brz mv a sub/a
        """
    self.build_tree(['a', 'sub/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a'])
    osutils.rename('a', 'sub/a')
    self.run_bzr_error(['^brz: ERROR: Could not move a => a: sub is not versioned\\.$'], 'mv a sub/a')
    self.assertPathDoesNotExist('a')
    self.assertPathExists('sub/a')