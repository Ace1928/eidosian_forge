from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test__preload_handles_partially_evaluated_inventory(self):
    new_inv = self.make_basic_utf8_inventory()
    ie = new_inv.get_entry(new_inv.root_id)
    self.assertIs(None, ie._children)
    self.assertEqual(['dir-€', 'fïle'], sorted(ie.children.keys()))
    self.assertEqual(['dir-€', 'fïle'], sorted(ie._children.keys()))
    new_inv._preload_cache()
    self.assertEqual(['dir-€', 'fïle'], sorted(ie._children.keys()))
    ie_dir = new_inv.get_entry(b'dirid')
    self.assertEqual(['chïld'], sorted(ie_dir._children.keys()))