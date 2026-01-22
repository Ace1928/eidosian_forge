from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def assert_Getitems(self, expected_fileids, inv, file_ids):
    self.assertEqual(sorted(expected_fileids), sorted([ie.file_id for ie in inv._getitems(file_ids)]))