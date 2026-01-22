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
def check_thorough_reweave_missing_revision(self, aBzrDir, reconcile, **kwargs):
    repo = aBzrDir.open_repository()
    if not repo.has_revision(b'missing'):
        expected_inconsistent_parents = 0
    else:
        expected_inconsistent_parents = 1
    reconciler = reconcile(**kwargs)
    self.assertEqual(expected_inconsistent_parents, reconciler.inconsistent_parents)
    self.assertEqual(1, reconciler.garbage_inventories)
    repo = aBzrDir.open_repository()
    self.check_missing_was_removed(repo)
    self.assertFalse(repo.has_revision(b'missing'))