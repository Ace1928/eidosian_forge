from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_non_directory_children(self):
    """Test path2id when a parent directory has no children"""
    inv = inventory.Inventory(b'tree-root')
    inv.add(self.make_file(b'file-id', 'file', b'tree-root'))
    inv.add(self.make_link(b'link-id', 'link', b'tree-root'))
    self.assertIs(None, inv.path2id('file/subfile'))
    self.assertIs(None, inv.path2id('link/subfile'))