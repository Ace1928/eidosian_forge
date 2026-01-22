from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_unicode_file_id(self):
    inv = self.get_empty_inventory()
    dir1 = inventory.InventoryDirectory(b'dirid', 'dir1', inv.root.file_id)
    dir1.file_id = 'dirid'
    dir1.revision = b'result'
    delta = [(None, 'dir1', dir1.file_id, dir1)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)