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
def make_branch_with_disjoint_inventory_and_revision(self):
    """a repo with separate packs for a revisions Revision and Inventory.

        There will be one pack file that holds the Revision content, and one
        for the Inventory content.

        :return: (repository,
                  pack_name_with_rev_A_Revision,
                  pack_name_with_rev_A_Inventory,
                  pack_name_with_rev_C_content)
        """
    b_source = self.make_abc_branch()
    b_base = b_source.controldir.sprout('base', revision_id=b'A').open_branch()
    b_stacked = b_base.controldir.sprout('stacked', stacked=True).open_branch()
    b_stacked.lock_write()
    self.addCleanup(b_stacked.unlock)
    b_stacked.fetch(b_source, b'B')
    repo_not_stacked = b_stacked.controldir.open_repository()
    repo_not_stacked.lock_write()
    self.addCleanup(repo_not_stacked.unlock)
    self.assertEqual([(b'A',), (b'B',)], sorted(repo_not_stacked.inventories.keys()))
    self.assertEqual([(b'B',)], sorted(repo_not_stacked.revisions.keys()))
    stacked_pack_names = repo_not_stacked._pack_collection.names()
    for name in stacked_pack_names:
        pack = repo_not_stacked._pack_collection.get_pack_by_name(name)
        keys = [n[1] for n in pack.inventory_index.iter_all_entries()]
        if (b'A',) in keys:
            inv_a_pack_name = name
            break
    else:
        self.fail("Could not find pack containing A's inventory")
    repo_not_stacked.fetch(b_source.repository, b'A')
    self.assertEqual([(b'A',), (b'B',)], sorted(repo_not_stacked.revisions.keys()))
    new_pack_names = set(repo_not_stacked._pack_collection.names())
    rev_a_pack_names = new_pack_names.difference(stacked_pack_names)
    self.assertEqual(1, len(rev_a_pack_names))
    rev_a_pack_name = list(rev_a_pack_names)[0]
    repo_not_stacked.fetch(b_source.repository, b'C')
    rev_c_pack_names = set(repo_not_stacked._pack_collection.names())
    rev_c_pack_names = rev_c_pack_names.difference(new_pack_names)
    self.assertEqual(1, len(rev_c_pack_names))
    rev_c_pack_name = list(rev_c_pack_names)[0]
    return (repo_not_stacked, rev_a_pack_name, inv_a_pack_name, rev_c_pack_name)