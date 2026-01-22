from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_different_repos_not_equal(self):
    """Repositories at different locations are not the same."""
    repo_one = self.make_repository('one')
    repo_two = self.make_repository('two')
    self.assertDifferentRepo(repo_one, repo_two)