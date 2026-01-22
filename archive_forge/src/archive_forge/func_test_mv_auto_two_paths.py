import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_auto_two_paths(self):
    self.make_abcd_tree()
    out, err = self.run_bzr('mv --auto tree tree2', retcode=3)
    self.assertEqual('brz: ERROR: Only one path may be specified to --auto.\n', err)