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
def test_autopack_reloads_and_stops(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
    orig_execute = packs._execute_pack_operations

    def _munged_execute_pack_ops(*args, **kwargs):
        tree.branch.repository.pack()
        return orig_execute(*args, **kwargs)
    packs._execute_pack_operations = _munged_execute_pack_ops
    packs._max_pack_count = lambda x: 1
    packs.pack_distribution = lambda x: [10]
    self.assertFalse(packs.autopack())
    self.assertEqual(1, len(packs.names()))
    self.assertEqual(tree.branch.repository._pack_collection.names(), packs.names())