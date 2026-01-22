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
def test_find_controldirs_evaluate(self):

    def evaluate(bzrdir):
        try:
            repo = bzrdir.open_repository()
        except errors.NoRepositoryPresent:
            return (True, bzrdir.root_transport.base)
        else:
            return (False, bzrdir.root_transport.base)
    foo, bar, baz = self.make_foo_bar_baz()
    t = self.get_transport()
    self.assertEqual([baz.root_transport.base, foo.root_transport.base], list(bzrdir.BzrDir.find_controldirs(t, evaluate=evaluate)))