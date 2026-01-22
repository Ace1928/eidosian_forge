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
def test_inventories_use_chk_map_with_parent_base_dict(self):
    tree = self.make_branch_and_memory_tree('repo', format='2a')
    tree.lock_write()
    tree.add([''], ids=[b'TREE_ROOT'])
    revid = tree.commit('foo')
    tree.unlock()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    inv = tree.branch.repository.get_inventory(revid)
    self.assertNotEqual(None, inv.parent_id_basename_to_file_id)
    inv.parent_id_basename_to_file_id._ensure_root()
    inv.id_to_entry._ensure_root()
    self.assertEqual(65536, inv.id_to_entry._root_node.maximum_size)
    self.assertEqual(65536, inv.parent_id_basename_to_file_id._root_node.maximum_size)