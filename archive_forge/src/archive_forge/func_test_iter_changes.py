from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_iter_changes(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
    inv.get_entry(b'fileid').revision = b'filerev'
    inv.get_entry(b'fileid').executable = True
    inv.get_entry(b'fileid').text_sha1 = b'ffff'
    inv.get_entry(b'fileid').text_size = 1
    inv2 = Inventory()
    inv2.revision_id = b'revid2'
    inv2.root.revision = b'rootrev'
    inv2.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
    inv2.get_entry(b'fileid').revision = b'filerev2'
    inv2.get_entry(b'fileid').executable = False
    inv2.get_entry(b'fileid').text_sha1 = b'bbbb'
    inv2.get_entry(b'fileid').text_size = 2
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    inv_1 = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
    chk_inv2 = CHKInventory.from_inventory(chk_bytes, inv2)
    lines = chk_inv2.to_lines()
    inv_2 = CHKInventory.deserialise(chk_bytes, lines, (b'revid2',))
    self.assertEqual([(b'fileid', ('file', 'file'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('file', 'file'), ('file', 'file'), (False, True))], list(inv_1.iter_changes(inv_2)))