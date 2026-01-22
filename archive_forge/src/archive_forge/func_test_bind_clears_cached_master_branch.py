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
def test_bind_clears_cached_master_branch(self):
    """b.bind clears any cached value of b.get_master_branch."""
    master1 = self.make_branch('master1')
    master2 = self.make_branch('master2')
    branch = self.make_branch('branch')
    try:
        branch.bind(master1)
    except _mod_branch.BindingUnsupported:
        raise tests.TestNotApplicable('Format does not support binding')
    self.addCleanup(branch.lock_write().unlock)
    self.assertNotEqual(None, branch.get_master_branch())
    branch.bind(master2)
    self.assertEqual('.', urlutils.relative_url(self.get_url('master2'), branch.get_master_branch().base))