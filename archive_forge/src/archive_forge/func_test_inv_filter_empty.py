from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_inv_filter_empty(self):
    inv = self.prepare_inv_with_nested_dirs()
    new_inv = inv.filter([])
    self.assertEqual([('', b'tree-root')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])