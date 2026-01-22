from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_iter_just_entries(self):
    inv = self.prepare_inv_with_nested_dirs()
    self.assertEqual([b'a-id', b'bye-id', b'doc-id', b'hello-id', b'makefile-id', b'src-id', b'sub-id', b'tree-root', b'zz-id', b'zzc-id'], sorted([ie.file_id for ie in inv.iter_just_entries()]))