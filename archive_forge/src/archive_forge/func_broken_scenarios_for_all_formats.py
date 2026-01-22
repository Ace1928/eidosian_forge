from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def broken_scenarios_for_all_formats():
    format_scenarios = all_repository_vf_format_scenarios()
    broken_scenarios = [(s.__name__, {'scenario_class': s}) for s in all_broken_scenario_classes]
    return multiply_scenarios(format_scenarios, broken_scenarios)