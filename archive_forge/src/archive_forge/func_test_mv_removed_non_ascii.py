import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_removed_non_ascii(self):
    """Clear error on mv of a removed non-ascii file, see lp:898541"""
    self.requireFeature(UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['§'])
    tree.add(['§'])
    tree.commit('Adding §')
    os.remove('§')
    out, err = self.run_bzr_error(['Could not rename', 'not exist'], ['mv', '§', 'b'])