from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_inv_filter_files_and_dirs(self):
    inv = self.prepare_inv_with_nested_dirs()
    new_inv = inv.filter([b'makefile-id', b'src-id'])
    self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('src', b'src-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('src/zz.c', b'zzc-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])