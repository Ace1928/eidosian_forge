import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file(self):
    """Test brz mv original_file to moved_file.

        Tests if a file which has allready been moved by an external tool,
        is handled correctly by brz mv.
        Setup: a is in the working tree, b does not exist.
        User does: mv a b; brz mv a b
        """
    self.build_tree(['a'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a'])
    osutils.rename('a', 'b')
    self.run_bzr('mv a b')
    self.assertMoved('a', 'b')