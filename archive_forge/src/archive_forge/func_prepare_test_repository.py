from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def prepare_test_repository(self):
    """Prepare a repository to test with from the test scenario.

        :return: A repository, and the scenario instance.
        """
    scenario = self.scenario_class(self)
    repo = self.make_populated_repository(scenario.populate_repository)
    self.require_repo_suffers_text_parent_corruption(repo)
    return (repo, scenario)