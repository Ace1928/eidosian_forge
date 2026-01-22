from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_duplicate_file_id(self):
    error = DuplicateFileId('a_file_id', 'foo')
    self.assertEqualDiff('File id {a_file_id} already exists in inventory as foo', str(error))