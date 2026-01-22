import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_get_missing_parent_inventories(self):
    """A stacked repo with a single revision and inventory (no parent
        inventory) in it must have all the texts in its inventory (even if not
        changed w.r.t. to the absent parent), otherwise it will report missing
        texts/parent inventory.

        The core of this test is that a file was changed in rev-1, but in a
        stacked repo that only has rev-2
        """
    trunk_repo = self.make_stackable_repo()
    self.make_first_commit(trunk_repo)
    trunk_repo.lock_read()
    self.addCleanup(trunk_repo.unlock)
    branch_repo = self.make_new_commit_in_new_repo(trunk_repo, parents=[b'rev-1'])
    inv = branch_repo.get_inventory(b'rev-2')
    repo = self.make_stackable_repo('stacked')
    repo.lock_write()
    repo.start_write_group()
    repo.add_inventory(b'rev-2', inv, [b'rev-1'])
    repo.revisions.insert_record_stream(branch_repo.revisions.get_record_stream([(b'rev-2',)], 'unordered', False))
    self.assertEqual(set(), repo.inventories.get_missing_compression_parent_keys())
    self.assertEqual({('inventories', b'rev-1')}, repo.get_missing_parent_inventories())
    reopened_repo = self.reopen_repo_and_resume_write_group(repo)
    self.assertEqual({('inventories', b'rev-1')}, reopened_repo.get_missing_parent_inventories())
    reopened_repo.inventories.insert_record_stream(branch_repo.inventories.get_record_stream([(b'rev-1',)], 'unordered', False))
    self.assertEqual(set(), reopened_repo.get_missing_parent_inventories())
    reopened_repo.abort_write_group()