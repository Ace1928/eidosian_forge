from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_locks_repo_too(self):
    branch = self.make_branch('b')
    branch = branch.controldir.open_branch()
    branch.lock_write()
    try:
        self.assertTrue(branch.repository.is_write_locked())
        if not branch.repository.get_physical_lock_status():
            return
        new_repo = branch.controldir.open_repository()
        self.assertRaises(errors.LockContention, new_repo.lock_write)
        branch.repository.lock_write()
        branch.repository.unlock()
    finally:
        branch.unlock()