from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_symlink_detect_changes(self):
    left = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
    left.symlink_target = 'foo'
    right = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
    right.symlink_target = 'foo'
    self.assertEqual((False, False), left.detect_changes(right))
    self.assertEqual((False, False), right.detect_changes(left))
    left.symlink_target = 'different'
    self.assertEqual((True, False), left.detect_changes(right))
    self.assertEqual((True, False), right.detect_changes(left))