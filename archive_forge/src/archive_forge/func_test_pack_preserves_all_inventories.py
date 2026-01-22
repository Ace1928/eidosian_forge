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
def test_pack_preserves_all_inventories(self):
    format = self.get_format()
    builder = self.make_branch_builder('source', format=format)
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot(None, [('add', ('file', b'file-id', 'file', b'B content\n'))], revision_id=b'B-id')
    builder.build_snapshot(None, [('modify', ('file', b'C content\n'))], revision_id=b'C-id')
    builder.finish_series()
    b = builder.get_branch()
    b.lock_read()
    self.addCleanup(b.unlock)
    repo = self.make_repository('repo', shared=True, format=format)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.fetch(b.repository, revision_id=b'B-id')
    inv = next(b.repository.iter_inventories([b'C-id']))
    repo.start_write_group()
    repo.add_inventory(b'C-id', inv, [b'B-id'])
    repo.commit_write_group()
    self.assertEqual([(b'A-id',), (b'B-id',), (b'C-id',)], sorted(repo.inventories.keys()))
    repo.pack()
    self.assertEqual([(b'A-id',), (b'B-id',), (b'C-id',)], sorted(repo.inventories.keys()))
    self.assertEqual(inv, next(repo.iter_inventories([b'C-id'])))