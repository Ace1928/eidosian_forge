from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_file_invalid_entry_name(self):
    self.assertRaises(InvalidEntryName, inventory.InventoryFile, b'123', 'a/hello.c', ROOT_ID)