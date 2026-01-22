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
def test_plan_pack_operations_creates_a_single_op(self):
    packs = self.get_packs()
    existing_packs = [(50, 'a'), (40, 'b'), (30, 'c'), (10, 'd'), (10, 'e'), (6, 'f'), (4, 'g')]
    distribution = packs.pack_distribution(150)
    pack_operations = packs.plan_autopack_combinations(existing_packs, distribution)
    self.assertEqual([[130, ['a', 'b', 'c', 'f', 'g']]], pack_operations)