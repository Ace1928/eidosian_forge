from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_renamed_dir_with_renamed_child(self):
    inv = self.get_empty_inventory()
    dir1 = inventory.InventoryDirectory(b'dir-id', 'dir1', inv.root.file_id)
    dir1.revision = b'basis'
    file1 = self.make_file_ie(b'file-id-1', 'name1', parent_id=b'dir-id')
    file2 = self.make_file_ie(b'file-id-2', 'name2', parent_id=b'dir-id')
    inv.add(dir1)
    inv.add(file1)
    inv.add(file2)
    dir2 = inventory.InventoryDirectory(b'dir-id', 'dir2', inv.root.file_id)
    dir2.revision = b'result'
    file2b = self.make_file_ie(b'file-id-2', 'name2', inv.root.file_id)
    delta = [('dir1', 'dir2', b'dir-id', dir2), ('dir1/name2', 'name2', b'file-id-2', file2b)]
    res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
    self.assertEqual(b'file-id-1', res_inv.path2id('dir2/name1'))
    self.assertEqual(None, res_inv.path2id('dir2/name2'))
    self.assertEqual(b'file-id-2', res_inv.path2id('name2'))