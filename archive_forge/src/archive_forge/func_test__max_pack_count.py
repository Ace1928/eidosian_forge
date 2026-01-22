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
def test__max_pack_count(self):
    """The maximum pack count is a function of the number of revisions."""
    packs = self.get_packs()
    self.assertEqual(1, packs._max_pack_count(0))
    self.assertEqual(1, packs._max_pack_count(1))
    self.assertEqual(2, packs._max_pack_count(2))
    self.assertEqual(3, packs._max_pack_count(3))
    self.assertEqual(4, packs._max_pack_count(4))
    self.assertEqual(5, packs._max_pack_count(5))
    self.assertEqual(6, packs._max_pack_count(6))
    self.assertEqual(7, packs._max_pack_count(7))
    self.assertEqual(8, packs._max_pack_count(8))
    self.assertEqual(9, packs._max_pack_count(9))
    self.assertEqual(1, packs._max_pack_count(10))
    self.assertEqual(2, packs._max_pack_count(11))
    self.assertEqual(10, packs._max_pack_count(19))
    self.assertEqual(2, packs._max_pack_count(20))
    self.assertEqual(3, packs._max_pack_count(21))
    self.assertEqual(25, packs._max_pack_count(112894))