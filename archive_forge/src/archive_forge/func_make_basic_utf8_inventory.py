from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def make_basic_utf8_inventory(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    root_id = inv.root.file_id
    inv.add(InventoryFile(b'fileid', 'fïle', root_id))
    inv.get_entry(b'fileid').revision = b'filerev'
    inv.get_entry(b'fileid').text_sha1 = b'ffff'
    inv.get_entry(b'fileid').text_size = 0
    inv.add(InventoryDirectory(b'dirid', 'dir-€', root_id))
    inv.get_entry(b'dirid').revision = b'dirrev'
    inv.add(InventoryFile(b'childid', 'chïld', b'dirid'))
    inv.get_entry(b'childid').revision = b'filerev'
    inv.get_entry(b'childid').text_sha1 = b'ffff'
    inv.get_entry(b'childid').text_size = 0
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    return CHKInventory.deserialise(chk_bytes, lines, (b'revid',))