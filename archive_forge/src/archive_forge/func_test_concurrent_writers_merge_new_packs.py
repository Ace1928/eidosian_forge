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
def test_concurrent_writers_merge_new_packs(self):
    format = self.get_format()
    self.make_repository('.', shared=True, format=format)
    r1 = repository.Repository.open('.')
    r2 = repository.Repository.open('.')
    with r1.lock_write():
        list(r1.all_revision_ids())
        with r2.lock_write():
            list(r2.all_revision_ids())
            r1.start_write_group()
            try:
                r2.start_write_group()
                try:
                    self._add_text(r1, b'fileidr1')
                    self._add_text(r2, b'fileidr2')
                except:
                    r2.abort_write_group()
                    raise
            except:
                r1.abort_write_group()
                raise
            try:
                r1.commit_write_group()
            except:
                r1.abort_write_group()
                r2.abort_write_group()
                raise
            r2.commit_write_group()
            r1._pack_collection.reset()
            r1._pack_collection.ensure_loaded()
            r2._pack_collection.ensure_loaded()
            self.assertEqual(r1._pack_collection.names(), r2._pack_collection.names())
            self.assertEqual(2, len(r1._pack_collection.names()))