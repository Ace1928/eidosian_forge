from breezy import branch, tests
def test_revision_history_with_lightweight_checkout(self):
    """With a repository branch lightweight checkout location."""
    self._build_branch()
    self.run_bzr('init-shared-repo repo')
    self.run_bzr('branch test repo/test')
    self.run_bzr('checkout --lightweight repo/test test-checkout')
    self._check_revision_history('test-checkout')