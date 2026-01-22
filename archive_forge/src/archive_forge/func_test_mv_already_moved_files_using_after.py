import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_files_using_after(self):
    """Test brz mv --after versioned_file to directory/unversioned_file.

        Tests if an existing versioned file can be forced to move to an
        existing unversioned file in some other directory using the --after
        option. With the result that bazaar considers
        directory/unversioned_file to be moved from versioned_file and
        versioned_file will become unversioned.

        Setup: a1, a2, sub are versioned and in the working tree,
               sub/a1, sub/a2 are in working tree.
        User does: mv a* sub; touch a1; touch a2; brz mv a1 a2 sub --after
        """
    self.build_tree(['a1', 'a2', 'sub/', 'sub/a1', 'sub/a2'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a1', 'a2', 'sub'])
    osutils.rename('a1', 'sub/a1')
    osutils.rename('a2', 'sub/a2')
    self.build_tree(['a1'])
    self.build_tree(['a2'])
    self.run_bzr('mv a1 a2 sub --after')
    self.assertPathExists('a1')
    self.assertPathExists('a2')
    self.assertPathExists('sub/a1')
    self.assertPathExists('sub/a2')
    self.assertInWorkingTree('sub/a1')
    self.assertInWorkingTree('sub/a2')