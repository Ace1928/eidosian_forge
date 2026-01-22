from breezy import repository
from breezy.tests.per_repository import TestCaseWithRepository
def test_refresh_data_write_locked(self):
    repo = self.make_repository('.')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.refresh_data()