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
def test_sprout_controldir_empty_under_shared_repo(self):
    dir = self.make_controldir('source')
    try:
        self.make_repository('target', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('format does not support shared repositories')
    target = dir.sprout(self.get_url('target/child'))
    self.assertRaises(errors.NoRepositoryPresent, target.open_repository)
    target.open_branch()
    try:
        target.open_workingtree()
    except errors.NoWorkingTree:
        repo = target.find_repository()
        self.assertFalse(repo.controldir._format.supports_workingtrees)