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
def test_reweave_inventory_without_revision(self):
    d_url = self.get_url('inventory_without_revision')
    d = BzrDir.open(d_url)
    repo = d.open_repository()
    if not repo._reconcile_does_inventory_gc:
        raise TestSkipped('Irrelevant test')
    self.checkUnreconciled(d, repo.reconcile())
    result = repo.reconcile(thorough=True)
    self.assertEqual(0, result.inconsistent_parents)
    self.assertEqual(1, result.garbage_inventories)
    self.check_missing_was_removed(repo)