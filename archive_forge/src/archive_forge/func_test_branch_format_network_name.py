import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def test_branch_format_network_name(self):
    br = self.make_branch('.')
    format = br._format
    network_name = format.network_name()
    self.assertIsInstance(network_name, bytes)
    if isinstance(format, remote.RemoteBranchFormat):
        br._ensure_real()
        real_branch = br._real_branch
        self.assertEqual(real_branch._format.network_name(), network_name)
    else:
        registry = _mod_branch.network_format_registry
        looked_up_format = registry.get(network_name)
        self.assertEqual(format.__class__, looked_up_format.__class__)