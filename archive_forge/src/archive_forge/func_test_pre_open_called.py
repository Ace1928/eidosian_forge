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
def test_pre_open_called(self):
    calls = []
    bzrdir.BzrDir.hooks.install_named_hook('pre_open', calls.append, None)
    transport = self.get_transport('foo')
    url = transport.base
    self.assertRaises(errors.NotBranchError, bzrdir.BzrDir.open, url)
    self.assertEqual([transport.base], [t.base for t in calls])