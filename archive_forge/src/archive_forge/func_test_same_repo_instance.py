from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_same_repo_instance(self):
    """A repository object is the same repository as itself."""
    repo = self.make_repository('.')
    self.assertSameRepo(repo, repo)