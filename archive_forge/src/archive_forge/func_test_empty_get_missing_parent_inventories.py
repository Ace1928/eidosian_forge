import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_empty_get_missing_parent_inventories(self):
    """A new write group has no missing parent inventories."""
    repo = self.make_repository('.')
    repo.lock_write()
    repo.start_write_group()
    try:
        self.assertEqual(set(), set(repo.get_missing_parent_inventories()))
    finally:
        repo.commit_write_group()
        repo.unlock()