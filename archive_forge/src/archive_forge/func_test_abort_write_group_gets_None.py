from breezy import errors
from breezy.tests import per_repository, test_server
from breezy.transport import memory
def test_abort_write_group_gets_None(self):
    repo = self.make_repository('.')
    repo.lock_write()
    repo.start_write_group()
    self.assertEqual(None, repo.abort_write_group())
    repo.unlock()