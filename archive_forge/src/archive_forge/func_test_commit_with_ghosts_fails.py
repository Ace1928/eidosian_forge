from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def test_commit_with_ghosts_fails(self):
    base_tree, stacked_tree = self.make_stacked_target()
    stacked_tree.set_parent_ids([stacked_tree.last_revision(), b'ghost-rev-id'])
    self.assertRaises(errors.BzrError, stacked_tree.commit, 'failed_commit')