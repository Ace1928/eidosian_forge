from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_illegal(self):
    inv = self.make_init_inventory()
    self.assertRaises(errors.InconsistentDelta, inv.create_by_apply_delta, [(None, 'a', b'id-1', self.make_file(b'id-1', 'a', b'tree-root')), (None, 'b', b'id-1', self.make_file(b'id-1', 'b', b'tree-root'))], b'new-rev-1')