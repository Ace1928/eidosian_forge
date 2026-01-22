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
def test_register_inter_repository_class(self):
    dummy_a = DummyRepository()
    dummy_a._format = RepositoryFormat()
    dummy_b = DummyRepository()
    dummy_b._format = RepositoryFormat()
    repo = self.make_repository('.')
    dummy_a._serializer = repo._serializer
    dummy_a._format.supports_tree_reference = repo._format.supports_tree_reference
    dummy_a._format.rich_root_data = repo._format.rich_root_data
    dummy_a._format.supports_full_versioned_files = repo._format.supports_full_versioned_files
    dummy_b._serializer = repo._serializer
    dummy_b._format.supports_tree_reference = repo._format.supports_tree_reference
    dummy_b._format.rich_root_data = repo._format.rich_root_data
    dummy_b._format.supports_full_versioned_files = repo._format.supports_full_versioned_files
    repository.InterRepository.register_optimiser(InterDummy)
    try:
        self.assertFalse(InterDummy.is_compatible(dummy_a, repo))
        self.assertGetsDefaultInterRepository(dummy_a, repo)
        self.assertTrue(InterDummy.is_compatible(dummy_a, dummy_b))
        inter_repo = repository.InterRepository.get(dummy_a, dummy_b)
        self.assertEqual(InterDummy, inter_repo.__class__)
        self.assertEqual(dummy_a, inter_repo.source)
        self.assertEqual(dummy_b, inter_repo.target)
    finally:
        repository.InterRepository.unregister_optimiser(InterDummy)
    self.assertGetsDefaultInterRepository(dummy_a, dummy_b)