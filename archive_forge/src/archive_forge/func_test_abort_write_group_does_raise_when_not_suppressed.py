from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_abort_write_group_does_raise_when_not_suppressed(self):
    self.vfs_transport_factory = memory.MemoryServer
    repo = self.make_repository('repo', format=self.get_format())
    token = self._lock_write(repo).repository_token
    repo.start_write_group()
    self.get_transport('').rename('repo', 'foo')
    self.assertRaises(Exception, repo.abort_write_group)
    if token is not None:
        repo.leave_lock_in_place()