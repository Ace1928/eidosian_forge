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
def test_open_branch(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    t = self.get_transport()
    made_control = self.bzrdir_format.initialize(t.base)
    made_control.create_repository()
    made_branch = made_control.create_branch()
    opened_branch = made_control.open_branch()
    self.assertEqual(made_control, opened_branch.controldir)
    self.assertIsInstance(opened_branch, made_branch.__class__)
    self.assertIsInstance(opened_branch._format, made_branch._format.__class__)