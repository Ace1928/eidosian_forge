from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_attribute_text_store(self):
    """Test the existence of the texts attribute."""
    tree = self.make_branch_and_tree('tree')
    repo = tree.branch.repository
    self.assertIsInstance(repo.texts, versionedfile.VersionedFiles)