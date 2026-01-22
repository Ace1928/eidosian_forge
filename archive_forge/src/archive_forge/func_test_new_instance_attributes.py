from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def test_new_instance_attributes(self):
    upload_transport = self.get_transport('upload')
    pack_transport = self.get_transport('pack')
    index_transport = self.get_transport('index')
    upload_transport.mkdir('.')
    collection = pack_repo.RepositoryPackCollection(repo=None, transport=self.get_transport('.'), index_transport=index_transport, upload_transport=upload_transport, pack_transport=pack_transport, index_builder_class=BTreeBuilder, index_class=BTreeGraphIndex, use_chk_index=False)
    pack = pack_repo.NewPack(collection)
    self.addCleanup(pack.abort)
    self.assertIsInstance(pack.revision_index, BTreeBuilder)
    self.assertIsInstance(pack.inventory_index, BTreeBuilder)
    self.assertIsInstance(pack._hash, type(osutils.md5()))
    self.assertTrue(pack.upload_transport is upload_transport)
    self.assertTrue(pack.index_transport is index_transport)
    self.assertTrue(pack.pack_transport is pack_transport)
    self.assertEqual(None, pack.index_sizes)
    self.assertEqual(20, len(pack.random_name))
    self.assertIsInstance(pack.random_name, str)
    self.assertIsInstance(pack.start_time, float)