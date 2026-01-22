from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_symlink2_entry_to_bytes(self):
    inv = CHKInventory(None)
    ie = inventory.InventoryLink(b'link-id', 'linkΩname', b'parent-id')
    ie.revision = b'link-rev-id'
    ie.symlink_target = 'target/Ωpath'
    bytes = inv._entry_to_bytes(ie)
    self.assertEqual(b'symlink: link-id\nparent-id\nlink\xce\xa9name\nlink-rev-id\ntarget/\xce\xa9path', bytes)
    ie2 = inv._bytes_to_entry(bytes)
    self.assertEqual(ie, ie2)
    self.assertIsInstance(ie2.name, str)
    self.assertIsInstance(ie2.symlink_target, str)
    self.assertEqual((b'link\xce\xa9name', b'link-id', b'link-rev-id'), inv._bytes_to_utf8name_key(bytes))