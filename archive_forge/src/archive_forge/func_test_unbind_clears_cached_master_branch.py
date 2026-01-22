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
def test_unbind_clears_cached_master_branch(self):
    """b.unbind clears any cached value of b.get_master_branch."""
    master = self.make_branch('master')
    branch = self.make_branch('branch')
    try:
        branch.bind(master)
    except _mod_branch.BindingUnsupported:
        raise tests.TestNotApplicable('Format does not support binding')
    self.addCleanup(branch.lock_write().unlock)
    self.assertNotEqual(None, branch.get_master_branch())
    branch.unbind()
    self.assertEqual(None, branch.get_master_branch())