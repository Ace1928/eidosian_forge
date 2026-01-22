from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def test__generate_text_key_index(self):
    """Test that the generated text key index has all entries."""
    repo, scenario = self.prepare_test_repository()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual(scenario.repository_text_key_index(), repo._generate_text_key_index())