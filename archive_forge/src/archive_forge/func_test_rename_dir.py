from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_rename_dir(self):
    inv = self.get_empty_inventory()
    dir1 = inventory.InventoryDirectory(b'dir-id', 'dir1', inv.root.file_id)
    dir1.revision = b'basis'
    file1 = self.make_file_ie(parent_id=b'dir-id')
    inv.add(dir1)
    inv.add(file1)
    dir2 = inventory.InventoryDirectory(b'dir-id', 'dir2', inv.root.file_id)
    dir2.revision = b'result'
    delta = [('dir1', 'dir2', b'dir-id', dir2)]
    res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
    self.assertEqual(b'file-id', res_inv.path2id('dir2/name'))