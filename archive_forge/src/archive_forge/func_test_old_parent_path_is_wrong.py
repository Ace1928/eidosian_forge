from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_old_parent_path_is_wrong(self):
    inv = self.get_empty_inventory()
    parent1 = inventory.InventoryDirectory(b'p-1', 'dir', inv.root.file_id)
    parent1.revision = b'result'
    parent2 = inventory.InventoryDirectory(b'p-2', 'dir2', inv.root.file_id)
    parent2.revision = b'result'
    file1 = inventory.InventoryFile(b'id', 'path', b'p-2')
    file1.revision = b'result'
    file1.text_size = 0
    file1.text_sha1 = b''
    inv.add(parent1)
    inv.add(parent2)
    inv.add(file1)
    delta = [('dir/path', None, b'id', None)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)