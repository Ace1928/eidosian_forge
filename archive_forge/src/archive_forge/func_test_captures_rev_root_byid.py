from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_captures_rev_root_byid(self):
    inv = Inventory()
    inv.revision_id = b'foo'
    inv.root.revision = b'bar'
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    self.assertEqual([b'chkinventory:\n', b'revision_id: foo\n', b'root_id: TREE_ROOT\n', b'parent_id_basename_to_file_id: sha1:eb23f0ad4b07f48e88c76d4c94292be57fb2785f\n', b'id_to_entry: sha1:debfe920f1f10e7929260f0534ac9a24d7aabbb4\n'], lines)
    chk_inv = CHKInventory.deserialise(chk_bytes, lines, (b'foo',))
    self.assertEqual(b'plain', chk_inv._search_key_name)