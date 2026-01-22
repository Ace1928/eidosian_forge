from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_mismatched_new_path_entry_None(self):
    inv = self.get_empty_inventory()
    delta = [(None, 'path', b'id', None)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)