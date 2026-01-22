from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_commit_write_group_does_not_error(self):
    repo = self.make_repository('.')
    repo.lock_write()
    repo.start_write_group()
    repo.commit_write_group()
    repo.unlock()