from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_iter_entries_by_dir(self):
    inv = self.prepare_inv_with_nested_dirs()
    self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir()])
    self.assertEqual([('', b'tree-root'), ('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('src', b'src-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'a-id', b'zzc-id', b'doc-id', b'tree-root', b'hello-id', b'bye-id', b'zz-id', b'src-id', b'makefile-id', b'sub-id'))])
    self.assertEqual([('Makefile', b'makefile-id'), ('doc', b'doc-id'), ('zz', b'zz-id'), ('src/bye.c', b'bye-id'), ('src/hello.c', b'hello-id'), ('src/zz.c', b'zzc-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'a-id', b'zzc-id', b'doc-id', b'hello-id', b'bye-id', b'zz-id', b'makefile-id'))])
    self.assertEqual([('Makefile', b'makefile-id'), ('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id', b'makefile-id'))])
    self.assertEqual([('Makefile', b'makefile-id'), ('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id', b'makefile-id'))])
    self.assertEqual([('src/bye.c', b'bye-id')], [(path, ie.file_id) for path, ie in inv.iter_entries_by_dir(specific_file_ids=(b'bye-id',))])