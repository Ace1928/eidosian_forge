from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_same_bzrdir_different_control_files_not_equal(self):
    """Repositories in the same bzrdir, but with different control files,
        are not the same.

        This can happens e.g. when upgrading a repository.  This test mimics how
        CopyConverter creates a second repository in one bzrdir.
        """
    repo = self.make_repository('repo')
    repo.control_transport.copy_tree('.', '../repository.backup')
    backup_transport = repo.control_transport.clone('../repository.backup')
    if not repo._format.supports_overriding_transport:
        raise TestNotApplicable("remote repositories don't support overriding transport")
    backup_repo = repo._format.open(repo.controldir, _override_transport=backup_transport)
    self.assertDifferentRepo(repo, backup_repo)