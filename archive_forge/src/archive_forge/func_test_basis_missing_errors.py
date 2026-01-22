from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_basis_missing_errors(self):
    repo = self._get_repo_in_write_group()
    try:
        self.assertRaises(errors.NoSuchRevision, repo.add_inventory_by_delta, 'missing-revision', [], 'new-revision', ['missing-revision'])
    finally:
        repo.abort_write_group()