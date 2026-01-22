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
def test_nicks_bzr(self):
    """Test the behaviour of branch nicks specific to bzr branches.

        Nicknames are implicitly the name of the branch's directory, unless an
        explicit nickname is set.  That is, an explicit nickname always
        overrides the implicit one.

        """
    t = self.get_transport()
    branch = self.make_branch('bzr.dev')
    if not branch.repository._format.supports_storing_branch_nick:
        raise tests.TestNotApplicable('not a bzr branch format')
    self.assertEqual(branch.nick, 'bzr.dev')
    t.move('bzr.dev', 'bzr.ab')
    branch = _mod_branch.Branch.open(self.get_url('bzr.ab'))
    self.assertEqual(branch.nick, 'bzr.ab')
    branch.nick = "Aaron's branch"
    if not isinstance(branch, remote.RemoteBranch):
        self.assertTrue(branch._transport.has('branch.conf'))
    self.assertEqual(branch.nick, "Aaron's branch")
    t.move('bzr.ab', 'integration')
    branch = _mod_branch.Branch.open(self.get_url('integration'))
    self.assertEqual(branch.nick, "Aaron's branch")
    branch.nick = 'ሴ'
    self.assertEqual(branch.nick, 'ሴ')