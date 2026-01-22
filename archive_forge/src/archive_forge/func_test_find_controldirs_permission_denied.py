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
def test_find_controldirs_permission_denied(self):
    foo, bar, baz = self.make_foo_bar_baz()
    t = self.get_transport()
    path_filter_server, path_filter_transport = self.make_fake_permission_denied_transport(t, ['foo'])
    self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_controldirs(path_filter_transport))
    smart_transport = self.make_smart_server('.', backing_server=path_filter_server)
    self.assertBranchUrlsEndWith('/baz/', bzrdir.BzrDir.find_controldirs(smart_transport))