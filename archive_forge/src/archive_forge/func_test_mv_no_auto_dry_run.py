import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_no_auto_dry_run(self):
    self.make_abcd_tree()
    out, err = self.run_bzr('mv c d --dry-run', working_dir='tree', retcode=3)
    self.assertEqual('brz: ERROR: --dry-run requires --auto.\n', err)