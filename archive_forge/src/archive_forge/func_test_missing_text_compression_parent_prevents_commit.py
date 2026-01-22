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
def test_missing_text_compression_parent_prevents_commit(self):
    repo = self.make_write_ready_repo()
    key = ('some', 'junk')
    repo.texts._index._missing_compression_parents.add(key)
    self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
    e = self.assertRaises(errors.BzrCheckError, repo.commit_write_group)