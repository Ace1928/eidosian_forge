from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_not_in_write_group_errors(self):
    repo = self.make_repository('repository')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    self.assertRaises(AssertionError, repo.add_inventory_by_delta, 'missing-revision', [], 'new-revision', ['missing-revision'])