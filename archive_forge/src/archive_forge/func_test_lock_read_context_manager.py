from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_read_context_manager(self):
    branch = self.make_branch('b')
    self.assertFalse(branch.is_locked())
    with branch.lock_read():
        self.assertTrue(branch.is_locked())