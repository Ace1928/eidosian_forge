from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_populated_repository(self, factory):
    """Create a new repository populated by the given factory."""
    repo = self.make_repository('broken-repo')
    with repo.lock_write(), WriteGroup(repo):
        factory(repo)
        return repo