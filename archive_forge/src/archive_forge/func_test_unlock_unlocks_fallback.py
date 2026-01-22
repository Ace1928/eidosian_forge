from breezy import branch, errors
from breezy.tests.per_repository_reference import \
def test_unlock_unlocks_fallback(self):
    base = self.make_branch('base')
    stacked = self.make_branch('stacked')
    repo = stacked.repository
    stacked.set_stacked_on_url('../base')
    self.assertEqual(1, len(repo._fallback_repositories))
    fallback_repo = repo._fallback_repositories[0]
    self.assertFalse(repo.is_locked())
    self.assertFalse(fallback_repo.is_locked())
    repo.lock_read()
    self.assertTrue(repo.is_locked())
    self.assertTrue(fallback_repo.is_locked())
    repo.unlock()
    self.assertFalse(repo.is_locked())
    self.assertFalse(fallback_repo.is_locked())
    repo.lock_write()
    self.assertTrue(repo.is_locked())
    self.assertTrue(fallback_repo.is_locked())
    repo.unlock()
    self.assertFalse(repo.is_locked())
    self.assertFalse(fallback_repo.is_locked())