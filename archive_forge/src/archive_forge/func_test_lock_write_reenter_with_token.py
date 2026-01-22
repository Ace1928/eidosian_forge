from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_lock_write_reenter_with_token(self):
    branch = self.make_branch('b')
    with branch.lock_write() as lock:
        if lock.token is None:
            return
        branch.lock_write(token=lock.token)
        branch.unlock()
    new_branch = branch.controldir.open_branch()
    new_branch.lock_write()
    new_branch.unlock()