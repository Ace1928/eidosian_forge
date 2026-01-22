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
def test_upgrade_simple(self):
    """Upgrade simple v0.0.4 format to latest format"""
    eq = self.assertEqual
    self.build_tree_contents(_upgrade1_template)
    upgrade.upgrade('.')
    control = controldir.ControlDir.open('.')
    b = control.open_branch()
    self.assertIsInstance(control._format, bzrdir.BzrDirFormat.get_default_format().__class__)
    self.addCleanup(b.lock_read().unlock)
    rh = b._revision_history()
    eq(rh, [b'mbp@sourcefrog.net-20051004035611-176b16534b086b3c', b'mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd'])
    rt = b.repository.revision_tree(rh[0])
    foo_id = b'foo-20051004035605-91e788d1875603ae'
    with rt.lock_read():
        eq(rt.get_file_text('foo'), b'initial contents\n')
    rt = b.repository.revision_tree(rh[1])
    with rt.lock_read():
        eq(rt.get_file_text('foo'), b'new contents\n')
    backup_dir = 'backup.bzr.~1~'
    t = self.get_transport('.')
    t.stat(backup_dir)
    t.stat(backup_dir + '/README')
    t.stat(backup_dir + '/branch-format')
    t.stat(backup_dir + '/revision-history')
    t.stat(backup_dir + '/merged-patches')
    t.stat(backup_dir + '/pending-merged-patches')
    t.stat(backup_dir + '/pending-merges')
    t.stat(backup_dir + '/branch-name')
    t.stat(backup_dir + '/branch-lock')
    t.stat(backup_dir + '/inventory')
    t.stat(backup_dir + '/stat-cache')
    t.stat(backup_dir + '/text-store')
    t.stat(backup_dir + '/text-store/foo-20051004035611-1591048e9dc7c2d4.gz')
    t.stat(backup_dir + '/text-store/foo-20051004035756-4081373d897c3453.gz')
    t.stat(backup_dir + '/inventory-store/')
    t.stat(backup_dir + '/inventory-store/mbp@sourcefrog.net-20051004035611-176b16534b086b3c.gz')
    t.stat(backup_dir + '/inventory-store/mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd.gz')
    t.stat(backup_dir + '/revision-store/')
    t.stat(backup_dir + '/revision-store/mbp@sourcefrog.net-20051004035611-176b16534b086b3c.gz')
    t.stat(backup_dir + '/revision-store/mbp@sourcefrog.net-20051004035756-235f2b7dcdddd8dd.gz')