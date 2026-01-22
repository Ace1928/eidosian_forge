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
def test_get_default_inter_repository(self):
    dummy_a = DummyRepository()
    dummy_a._format = RepositoryFormat()
    dummy_a._format.supports_full_versioned_files = True
    dummy_a._format.rich_root_data = True
    dummy_b = DummyRepository()
    dummy_b._format = RepositoryFormat()
    dummy_b._format.supports_full_versioned_files = True
    dummy_b._format.rich_root_data = True
    self.assertGetsDefaultInterRepository(dummy_a, dummy_b)