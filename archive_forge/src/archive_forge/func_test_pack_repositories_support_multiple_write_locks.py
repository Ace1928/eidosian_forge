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
def test_pack_repositories_support_multiple_write_locks(self):
    format = self.get_format()
    self.make_repository('.', shared=True, format=format)
    r1 = repository.Repository.open('.')
    r2 = repository.Repository.open('.')
    r1.lock_write()
    self.addCleanup(r1.unlock)
    r2.lock_write()
    r2.unlock()