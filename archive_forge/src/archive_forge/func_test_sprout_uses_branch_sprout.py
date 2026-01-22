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
def test_sprout_uses_branch_sprout(self):
    """BzrDir.sprout calls Branch.sprout.

        Usually, BzrDir.sprout should delegate to the branch's sprout method
        for part of the work.  This allows the source branch to control the
        choice of format for the new branch.

        There are exceptions, but this tests avoids them:
          - if there's no branch in the source bzrdir,
          - or if the stacking has been requested and the format needs to be
            overridden to satisfy that.
        """
    t = self.get_transport('source')
    t.ensure_base()
    source_bzrdir = _TestBzrDirFormat().initialize_on_transport(t)
    self.assertEqual([], source_bzrdir.test_branch.calls)
    target_url = self.get_url('target')
    result = source_bzrdir.sprout(target_url, recurse='no')
    self.assertSubset(['sprout'], source_bzrdir.test_branch.calls)