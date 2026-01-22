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
def test_plan_pack_operations_2009_revisions_skip_all_packs(self):
    packs = self.get_packs()
    existing_packs = [(2000, 'big'), (9, 'medium')]
    pack_operations = packs.plan_autopack_combinations(existing_packs, [1000, 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    self.assertEqual([], pack_operations)