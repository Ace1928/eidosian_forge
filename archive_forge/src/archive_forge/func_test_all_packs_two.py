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
def test_all_packs_two(self):
    format = self.get_format()
    tree = self.make_branch_and_tree('.', format=format)
    tree.commit('start')
    tree.commit('continue')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    packs = tree.branch.repository._pack_collection
    packs.ensure_loaded()
    self.assertEqual([packs.get_pack_by_name(packs.names()[0]), packs.get_pack_by_name(packs.names()[1])], packs.all_packs())