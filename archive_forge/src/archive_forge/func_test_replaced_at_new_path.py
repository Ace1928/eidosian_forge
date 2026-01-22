from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_replaced_at_new_path(self):
    inv = self.get_empty_inventory()
    file1 = self.make_file_ie(file_id=b'id1', parent_id=inv.root.file_id)
    inv.add(file1)
    file2 = self.make_file_ie(file_id=b'id2', parent_id=inv.root.file_id)
    delta = [('name', None, b'id1', None), (None, 'name', b'id2', file2)]
    res_inv = self.apply_delta(self, inv, delta, invalid_delta=False)
    self.assertEqual(b'id2', res_inv.path2id('name'))