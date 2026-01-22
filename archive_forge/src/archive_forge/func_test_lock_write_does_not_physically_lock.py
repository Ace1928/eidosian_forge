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
def test_lock_write_does_not_physically_lock(self):
    repo = self.make_repository('.', format=self.get_format())
    repo.lock_write()
    self.addCleanup(repo.unlock)
    self.assertFalse(repo.get_physical_lock_status())