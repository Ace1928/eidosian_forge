import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
def test_upgrade_v6_to_meta_no_workingtree(self):
    self.build_tree_contents(_upgrade1_template)
    upgrade.upgrade('.', BzrDirFormat6())
    t = self.get_transport('.')
    t.delete('.bzr/pending-merges')
    t.delete('.bzr/inventory')
    self.assertFalse(t.has('.bzr/stat-cache'))
    t.delete_tree('backup.bzr.~1~')
    upgrade.upgrade('.', bzrdir.BzrDirMetaFormat1())
    control = controldir.ControlDir.open('.')
    self.assertFalse(control.has_workingtree())
    self.assertIsInstance(control._format, bzrdir.BzrDirMetaFormat1)
    b = control.open_branch()
    self.addCleanup(b.lock_read().unlock)
    self.assertEqual(b._revision_history(), [b'mbp@sourcefrog.net-20051004035611-176b16534b086b3c', b'mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd'])