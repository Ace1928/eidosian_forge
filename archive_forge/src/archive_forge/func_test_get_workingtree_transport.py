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
def test_get_workingtree_transport(self):
    dir = self.make_controldir('.')
    self.assertTrue(isinstance(dir.get_workingtree_transport(None), transport.Transport))
    anonymous_format = AnonymousTestWorkingTreeFormat()
    identifiable_format = IdentifiableTestWorkingTreeFormat()
    try:
        found_transport = dir.get_workingtree_transport(anonymous_format)
        self.assertRaises(errors.IncompatibleFormat, dir.get_workingtree_transport, identifiable_format)
    except errors.IncompatibleFormat:
        found_transport = dir.get_workingtree_transport(identifiable_format)
    self.assertTrue(isinstance(found_transport, transport.Transport))
    found_transport.list_dir('.')