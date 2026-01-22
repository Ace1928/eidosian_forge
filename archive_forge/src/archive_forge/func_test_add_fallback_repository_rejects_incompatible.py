from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def test_add_fallback_repository_rejects_incompatible(self):
    referring, fallback = self.make_repo_and_incompatible_fallback()
    self.assertRaises(errors.IncompatibleRepositories, referring.add_fallback_repository, fallback)