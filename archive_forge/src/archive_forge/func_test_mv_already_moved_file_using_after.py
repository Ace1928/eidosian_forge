import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file_using_after(self):
    """Test brz mv --after versioned_file to unversioned_file.

        Tests if an existing versioned file can be forced to move to an
        existing unversioned file using the --after option. With the result
        that bazaar considers the unversioned_file to be moved from
        versioned_file and versioned_file will become unversioned.
        Setup: a is in the working tree and b exists.
        User does: mv a b; touch a; brz mv a b --after
        Resulting in a => b and a is unknown.
        """
    self.build_tree(['a', 'b'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a'])
    osutils.rename('a', 'b')
    self.build_tree(['a'])
    self.run_bzr('mv a b --after')
    self.assertPathExists('a')
    self.assertNotInWorkingTree('a')
    self.assertPathExists('b')
    self.assertInWorkingTree('b')