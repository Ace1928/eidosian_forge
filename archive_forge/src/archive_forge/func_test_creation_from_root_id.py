from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_creation_from_root_id(self):
    inv = inventory.Inventory(root_id=b'tree-root')
    self.assertNotEqual(None, inv.root)
    self.assertEqual(b'tree-root', inv.root.file_id)