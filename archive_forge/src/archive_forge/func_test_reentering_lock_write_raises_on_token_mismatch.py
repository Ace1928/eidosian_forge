from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_reentering_lock_write_raises_on_token_mismatch(self):
    branch = self.make_branch('b')
    with branch.lock_write() as lock:
        if lock.token is None:
            return
        different_branch_token = lock.token + b'xxx'
        self.assertRaises(errors.TokenMismatch, branch.lock_write, token=different_branch_token)