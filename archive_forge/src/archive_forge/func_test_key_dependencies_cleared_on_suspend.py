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
def test_key_dependencies_cleared_on_suspend(self):
    source_repo, target_repo = self.create_source_and_target()
    target_repo.start_write_group()
    try:
        stream = source_repo.revisions.get_record_stream([(b'B-id',)], 'unordered', True)
        target_repo.revisions.insert_record_stream(stream)
        key_refs = target_repo.revisions._index._key_dependencies
        self.assertEqual([(b'B-id',)], sorted(key_refs.get_referrers()))
    finally:
        target_repo.suspend_write_group()
    self.assertEqual([], sorted(key_refs.get_referrers()))