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
def create_source_and_target(self):
    builder = self.make_branch_builder('source', format=self.get_format())
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id', b'ghost-id'], [], revision_id=b'B-id')
    builder.finish_series()
    repo = self.make_repository('target', format=self.get_format())
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    return (b.repository, repo)