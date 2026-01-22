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
def test_aborts_if_bad_parents_in_index(self):
    """Reconcile refuses to proceed if the revision index is wrong when
        checked against the revision texts, so that it does not generate broken
        data.

        Ideally reconcile would fix this, but until we implement that we just
        make sure we safely detect this problem.
        """
    repo = self.make_repo_with_extra_ghost_index()
    result = repo.reconcile(thorough=True)
    self.assertTrue(result.aborted, 'reconcile should have aborted due to bad parents.')