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
def test_open_containing_from_transport(self):
    self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('')))
    self.assertRaises(NotBranchError, bzrdir.BzrDir.open_containing_from_transport, _mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
    control = bzrdir.BzrDir.create(self.get_url())
    branch, relpath = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('')))
    self.assertEqual('', relpath)
    branch, relpath = bzrdir.BzrDir.open_containing_from_transport(_mod_transport.get_transport_from_url(self.get_readonly_url('g/p/q')))
    self.assertEqual('g/p/q', relpath)