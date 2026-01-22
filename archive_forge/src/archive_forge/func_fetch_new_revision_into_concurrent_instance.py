from breezy import errors, repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def fetch_new_revision_into_concurrent_instance(self, repo, token):
    """Create a new revision (revid 'new-rev') and fetch it into a
        concurrent instance of repo.
        """
    source = self.make_branch_and_memory_tree('source')
    source.lock_write()
    self.addCleanup(source.unlock)
    source.add([''], [b'root-id'])
    revid = source.commit('foo', rev_id=b'new-rev')
    repo.all_revision_ids()
    repo.revisions.keys()
    repo.inventories.keys()
    server_repo = repo.controldir.open_repository()
    try:
        server_repo.lock_write(token)
    except errors.TokenLockingNotSupported:
        self.skipTest('Cannot concurrently insert into repo format %r' % self.repository_format)
    try:
        server_repo.fetch(source.branch.repository, revid)
    finally:
        server_repo.unlock()