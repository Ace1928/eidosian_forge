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
def test_get_set_append_revisions_only(self):
    branch = self.make_branch('.')
    if branch._format.supports_set_append_revisions_only():
        branch.set_append_revisions_only(True)
        self.assertTrue(branch.get_append_revisions_only())
        branch.set_append_revisions_only(False)
        self.assertFalse(branch.get_append_revisions_only())
    else:
        self.assertRaises(errors.UpgradeRequired, branch.set_append_revisions_only, True)
        self.assertFalse(branch.get_append_revisions_only())