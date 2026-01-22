import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file_forcing_after(self):
    """Test brz mv versioned_file to unversioned_file.

        Tests if an attempt to move an existing versioned file to an existing
        unversioned file will fail, informing the user to use the --after
        option to force this.
        Setup: a is in the working tree, b not versioned.
        User does: mv a b; touch a; brz mv a b
        """
    self.build_tree(['a', 'b'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a'])
    osutils.rename('a', 'b')
    self.build_tree(['a'])
    self.run_bzr_error(['^brz: ERROR: Could not rename a => b because both files exist. \\(Use --after to tell brz about a rename that has already happened\\)$'], 'mv a b')
    self.assertPathExists('a')
    self.assertPathExists('b')