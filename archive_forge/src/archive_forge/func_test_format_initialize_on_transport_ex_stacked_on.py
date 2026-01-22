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
def test_format_initialize_on_transport_ex_stacked_on(self):
    trunk = self.make_branch('trunk', format='1.9')
    t = self.get_transport('stacked')
    old_fmt = controldir.format_registry.make_controldir('pack-0.92')
    repo_name = old_fmt.repository_format.network_name()
    repo, control, require_stacking, repo_policy = old_fmt.initialize_on_transport_ex(t, repo_format_name=repo_name, stacked_on='../trunk', stack_on_pwd=t.base)
    if repo is not None:
        self.assertTrue(repo.is_write_locked())
        self.addCleanup(repo.unlock)
    else:
        repo = control.open_repository()
    self.assertIsInstance(control, bzrdir.BzrDir)
    opened = bzrdir.BzrDir.open(t.base)
    if not isinstance(old_fmt, remote.RemoteBzrDirFormat):
        self.assertEqual(control._format.network_name(), old_fmt.network_name())
        self.assertEqual(control._format.network_name(), opened._format.network_name())
    self.assertEqual(control.__class__, opened.__class__)
    self.assertLength(1, repo._fallback_repositories)