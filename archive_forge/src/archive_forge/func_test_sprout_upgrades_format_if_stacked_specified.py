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
def test_sprout_upgrades_format_if_stacked_specified(self):
    child_branch, new_child_transport = self.prepare_default_stacking(child_format='pack-0.92')
    new_child = child_branch.controldir.sprout(new_child_transport.base, stacked=True)
    self.assertEqual(child_branch.controldir.root_transport.base, new_child.open_branch().get_stacked_on_url())
    repo = new_child.open_repository()
    self.assertTrue(repo._format.supports_external_lookups)
    self.assertFalse(repo.supports_rich_root())