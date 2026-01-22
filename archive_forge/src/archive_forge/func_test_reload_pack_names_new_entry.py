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
def test_reload_pack_names_new_entry(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo()
    names = packs.names()
    rev4 = tree.commit('four')
    new_names = tree.branch.repository._pack_collection.names()
    new_name = set(new_names).difference(names)
    self.assertEqual(1, len(new_name))
    new_name = new_name.pop()
    self.assertEqual(names, packs.names())
    self.assertTrue(packs.reload_pack_names())
    self.assertEqual(new_names, packs.names())
    self.assertEqual({rev4: (revs[-1],)}, r.get_parent_map([rev4]))
    self.assertFalse(packs.reload_pack_names())