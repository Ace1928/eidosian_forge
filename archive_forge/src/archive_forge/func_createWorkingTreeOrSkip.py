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
def createWorkingTreeOrSkip(self, a_controldir):
    """Create a working tree on a_controldir, or raise TestSkipped.

        A simple wrapper for create_workingtree that translates NotLocalUrl into
        TestSkipped.  Returns the newly created working tree.
        """
    try:
        return a_controldir.create_workingtree(revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False)
    except errors.NotLocalUrl:
        raise TestSkipped('cannot make working tree with transport %r' % a_controldir.transport)