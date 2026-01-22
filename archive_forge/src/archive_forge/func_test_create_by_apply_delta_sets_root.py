from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_create_by_apply_delta_sets_root(self):
    inv = Inventory()
    inv.root.revision = b'myrootrev'
    inv.revision_id = b'revid'
    chk_bytes = self.get_chk_bytes()
    base_inv = CHKInventory.from_inventory(chk_bytes, inv)
    inv.add_path('', 'directory', b'myrootid', None)
    inv.revision_id = b'expectedid'
    inv.root.revision = b'myrootrev'
    reference_inv = CHKInventory.from_inventory(chk_bytes, inv)
    delta = [('', None, base_inv.root.file_id, None), (None, '', b'myrootid', inv.root)]
    new_inv = base_inv.create_by_apply_delta(delta, b'expectedid')
    self.assertEqual(reference_inv.root, new_inv.root)