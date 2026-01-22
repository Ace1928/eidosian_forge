from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_dir_entry_to_bytes(self):
    inv = CHKInventory(None)
    ie = inventory.InventoryDirectory(b'dir-id', 'dirname', b'parent-id')
    ie.revision = b'dir-rev-id'
    bytes = inv._entry_to_bytes(ie)
    self.assertEqual(b'dir: dir-id\nparent-id\ndirname\ndir-rev-id', bytes)
    ie2 = inv._bytes_to_entry(bytes)
    self.assertEqual(ie, ie2)
    self.assertIsInstance(ie2.name, str)
    self.assertEqual((b'dirname', b'dir-id', b'dir-rev-id'), inv._bytes_to_utf8name_key(bytes))