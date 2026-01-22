from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_is_in_write_group(self):
    repo = self.make_repository('.')
    self.assertFalse(repo.is_in_write_group())
    repo.lock_write()
    repo.start_write_group()
    self.assertTrue(repo.is_in_write_group())
    repo.commit_write_group()
    self.assertFalse(repo.is_in_write_group())
    repo.start_write_group()
    self.assertTrue(repo.is_in_write_group())
    repo.abort_write_group()
    self.assertFalse(repo.is_in_write_group())
    repo.unlock()