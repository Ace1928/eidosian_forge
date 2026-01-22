from .. import branch, errors
from . import TestCaseWithTransport
def test_bad_repo_format(self):
    repo = self.make_repository('branch', shared=True, format='knit')
    a_branch = repo.controldir.create_branch()
    self.assertRaises(errors.RootNotRich, self.extract_in_checkout, a_branch)