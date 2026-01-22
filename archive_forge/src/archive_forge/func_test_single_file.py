from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_single_file(self):
    inv = self.make_simple_inventory()
    self.assertExpand([b'TREE_ROOT', b'top-id'], inv, [b'top-id'])