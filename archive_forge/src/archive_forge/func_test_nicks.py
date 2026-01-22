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
def test_nicks(self):
    """Test explicit and implicit branch nicknames.

        A nickname is always available, whether set explicitly or not.
        """
    t = self.get_transport()
    branch = self.make_branch('bzr.dev')
    self.assertIsInstance(branch.nick, str)
    branch.nick = "Aaron's branch"
    self.assertEqual(branch.nick, "Aaron's branch")
    branch.nick = 'ሴ'
    self.assertEqual(branch.nick, 'ሴ')