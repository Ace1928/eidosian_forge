from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_add_path_of_root(self):
    inv = inventory.Inventory(root_id=None)
    self.assertIs(None, inv.root)
    ie = inv.add_path('', 'directory', b'my-root')
    ie.revision = b'test-rev'
    self.assertEqual(b'my-root', ie.file_id)
    self.assertIs(ie, inv.root)