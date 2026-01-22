from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def make_referring(self, relpath, a_repository):
    """Get a new repository that refers to a_repository.

        :param relpath: The path to create the repository at.
        :param a_repository: A repository to refer to.
        """
    repo = self.make_repository(relpath)
    repo.add_fallback_repository(self.readonly_repository(a_repository))
    return repo