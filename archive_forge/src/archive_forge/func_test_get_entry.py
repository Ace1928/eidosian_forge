from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_get_entry(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
    inv.get_entry(b'fileid').revision = b'filerev'
    inv.get_entry(b'fileid').executable = True
    inv.get_entry(b'fileid').text_sha1 = b'ffff'
    inv.get_entry(b'fileid').text_size = 1
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
    root_entry = new_inv.get_entry(inv.root.file_id)
    file_entry = new_inv.get_entry(b'fileid')
    self.assertEqual('directory', root_entry.kind)
    self.assertEqual(inv.root.file_id, root_entry.file_id)
    self.assertEqual(inv.root.parent_id, root_entry.parent_id)
    self.assertEqual(inv.root.name, root_entry.name)
    self.assertEqual(b'rootrev', root_entry.revision)
    self.assertEqual('file', file_entry.kind)
    self.assertEqual(b'fileid', file_entry.file_id)
    self.assertEqual(inv.root.file_id, file_entry.parent_id)
    self.assertEqual('file', file_entry.name)
    self.assertEqual(b'filerev', file_entry.revision)
    self.assertEqual(b'ffff', file_entry.text_sha1)
    self.assertEqual(1, file_entry.text_size)
    self.assertEqual(True, file_entry.executable)
    self.assertRaises(errors.NoSuchId, new_inv.get_entry, 'missing')