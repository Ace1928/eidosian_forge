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
def test_create_null_workingtree(self):
    dir = self.make_controldir('dir1')
    dir.create_repository()
    dir.create_branch()
    try:
        wt = dir.create_workingtree(revision_id=_mod_revision.NULL_REVISION)
    except (errors.NotLocalUrl, errors.UnsupportedOperation):
        raise TestSkipped('cannot make working tree with transport %r' % dir.transport)
    self.assertEqual([], wt.get_parent_ids())