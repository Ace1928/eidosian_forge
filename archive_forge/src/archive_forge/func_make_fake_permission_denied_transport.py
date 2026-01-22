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
def make_fake_permission_denied_transport(self, transport, paths):
    """Create a transport that raises PermissionDenied for some paths."""

    def filter(path):
        if path in paths:
            raise errors.PermissionDenied(path)
        return path
    path_filter_server = pathfilter.PathFilteringServer(transport, filter)
    path_filter_server.start_server()
    self.addCleanup(path_filter_server.stop_server)
    path_filter_transport = pathfilter.PathFilteringTransport(path_filter_server, '.')
    return (path_filter_server, path_filter_transport)