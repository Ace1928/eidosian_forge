import threading
from dulwich.client import TCPGitClient
from dulwich.repo import Repo
from ...tests import TestCase, TestCaseWithTransport
from ...transport import transport_server_registry
from ..server import BzrBackend, BzrTCPGitServer
def test_fetch_from_non_git(self):
    wt = self.make_branch_and_tree('t', format='bzr')
    self.build_tree(['t/foo'])
    wt.add('foo')
    revid = wt.commit(message='some data')
    wt.branch.tags.set_tag('atag', revid)
    t = self.get_transport('t')
    port = self.start_server(t)
    c = TCPGitClient('localhost', port=port)
    gitrepo = Repo.init('gitrepo', mkdir=True)
    result = c.fetch('/', gitrepo)
    self.assertEqual(set(result.refs.keys()), {b'refs/tags/atag', b'HEAD'})