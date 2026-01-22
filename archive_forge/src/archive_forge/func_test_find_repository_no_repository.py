import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_find_repository_no_repository(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    url = self.get_vfs_only_url('subdir')
    transport.get_transport_from_url(self.get_vfs_only_url()).mkdir('subdir')
    made_control = self.bzrdir_format.initialize(self.get_url('subdir'))
    try:
        made_control.open_repository()
        return
    except errors.NoRepositoryPresent:
        pass
    made_control = controldir.ControlDir.open(self.get_readonly_url('subdir'))
    self.assertRaises(errors.NoRepositoryPresent, made_control.find_repository)