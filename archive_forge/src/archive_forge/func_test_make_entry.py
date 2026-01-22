from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_make_entry(self):
    self.assertIsInstance(inventory.make_entry('file', 'name', ROOT_ID), inventory.InventoryFile)
    self.assertIsInstance(inventory.make_entry('symlink', 'name', ROOT_ID), inventory.InventoryLink)
    self.assertIsInstance(inventory.make_entry('directory', 'name', ROOT_ID), inventory.InventoryDirectory)