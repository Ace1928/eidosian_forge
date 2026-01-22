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
def test_relative_branch_stacking_to_absolute(self):
    stack_on = self.make_branch('stack_on', format='1.6')
    stacked = self.make_branch('stack_on/stacked', format='1.6')
    policy = bzrdir.UseExistingRepository(stacked.repository, '.', self.get_readonly_url('stack_on'))
    policy.configure_branch(stacked)
    self.assertEqual(self.get_readonly_url('stack_on'), stacked.get_stacked_on_url())