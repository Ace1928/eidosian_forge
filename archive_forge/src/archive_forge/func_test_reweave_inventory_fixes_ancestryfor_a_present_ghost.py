import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
def test_reweave_inventory_fixes_ancestryfor_a_present_ghost(self):
    d = BzrDir.open(self.get_url('inventory_ghost_present'))
    repo = d.open_repository()
    m = MatchesAncestry(repo, b'ghost')
    if m.match([b'the_ghost', b'ghost']) is None:
        return
    self.assertThat([b'ghost'], m)
    reconciler = repo.reconcile()
    self.assertEqual(1, reconciler.inconsistent_parents)
    self.assertEqual(0, reconciler.garbage_inventories)
    repo = d.open_repository()
    repo.get_inventory(b'ghost')
    repo.get_inventory(b'the_ghost')
    self.assertThat([b'the_ghost', b'ghost'], MatchesAncestry(repo, b'ghost'))
    self.assertThat([b'the_ghost'], MatchesAncestry(repo, b'the_ghost'))