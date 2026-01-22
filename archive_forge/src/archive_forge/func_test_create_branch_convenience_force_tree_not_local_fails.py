import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_create_branch_convenience_force_tree_not_local_fails(self):
    format = controldir.format_registry.make_controldir('knit')
    self.assertRaises(errors.NotLocalUrl, bzrdir.BzrDir.create_branch_convenience, self.get_url('foo'), force_new_tree=True, format=format)
    t = self.get_transport()
    self.assertFalse(t.has('foo'))