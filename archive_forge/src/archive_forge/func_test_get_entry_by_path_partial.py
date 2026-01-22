from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_get_entry_by_path_partial(self):
    inv = inventory.Inventory(b'TREE_ROOT')
    inv.root.revision = b'revision'
    for args in [('src', 'directory', b'src-id'), ('doc', 'directory', b'doc-id'), ('src/hello.c', 'file'), ('src/bye.c', 'file', b'bye-id'), ('Makefile', 'file'), ('external', 'tree-reference', b'other-root')]:
        ie = inv.add_path(*args)
        ie.revision = b'revision'
        if args[1] == 'file':
            ie.text_sha1 = osutils.sha_string(b'content\n')
            ie.text_size = len(b'content\n')
        if args[1] == 'tree-reference':
            ie.reference_revision = b'reference'
    inv = self.inv_to_test_inv(inv)
    ie, resolved, remaining = inv.get_entry_by_path_partial('')
    self.assertEqual((ie.file_id, resolved, remaining), (b'TREE_ROOT', [], []))
    ie, resolved, remaining = inv.get_entry_by_path_partial('src')
    self.assertEqual((ie.file_id, resolved, remaining), (b'src-id', ['src'], []))
    ie, resolved, remaining = inv.get_entry_by_path_partial('src/bye.c')
    self.assertEqual((ie.file_id, resolved, remaining), (b'bye-id', ['src', 'bye.c'], []))
    ie, resolved, remaining = inv.get_entry_by_path_partial('external')
    self.assertEqual((ie.file_id, resolved, remaining), (b'other-root', ['external'], []))
    ie, resolved, remaining = inv.get_entry_by_path_partial('external/blah')
    self.assertEqual((ie.file_id, resolved, remaining), (b'other-root', ['external'], ['blah']))
    ie, resolved, remaining = inv.get_entry_by_path_partial('foo.c')
    self.assertEqual((ie, resolved, remaining), (None, None, None))