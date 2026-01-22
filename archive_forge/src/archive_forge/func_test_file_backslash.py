from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_file_backslash(self):
    file = inventory.InventoryFile(b'123', 'h\\ello.c', ROOT_ID)
    self.assertEqual(file.name, 'h\\ello.c')