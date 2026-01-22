import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_suspend_empty_initial_write_group(self):
    """Suspending a write group with no writes returns an empty token
        list.
        """
    self.require_suspendable_write_groups('Cannot test suspend on repo that does not support suspending')
    repo = self.make_write_locked_repo()
    repo.start_write_group()
    wg_tokens = repo.suspend_write_group()
    self.assertEqual([], wg_tokens)