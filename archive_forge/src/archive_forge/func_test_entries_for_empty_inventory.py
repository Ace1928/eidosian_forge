from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_entries_for_empty_inventory(self):
    """Test that entries() will not fail for an empty inventory"""
    inv = Inventory(root_id=None)
    self.assertEqual([], inv.entries())