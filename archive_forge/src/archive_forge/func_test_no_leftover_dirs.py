import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_no_leftover_dirs(self):
    if not self.bzrdir_format.colocated_branches:
        raise TestNotApplicable('format does not support colocated branches')
    branch = self.make_branch('.', format='development-colo')
    branch.controldir.create_branch(name='another-colocated-branch')
    self.assertEqual(branch.controldir.user_transport.list_dir('.'), ['.bzr'])