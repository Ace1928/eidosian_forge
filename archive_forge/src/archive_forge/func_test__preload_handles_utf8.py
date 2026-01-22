from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test__preload_handles_utf8(self):
    new_inv = self.make_basic_utf8_inventory()
    self.assertEqual({}, new_inv._fileid_to_entry_cache)
    self.assertFalse(new_inv._fully_cached)
    new_inv._preload_cache()
    self.assertEqual(sorted([new_inv.root_id, b'fileid', b'dirid', b'childid']), sorted(new_inv._fileid_to_entry_cache.keys()))
    ie_root = new_inv._fileid_to_entry_cache[new_inv.root_id]
    self.assertEqual(['dir-€', 'fïle'], sorted(ie_root._children.keys()))
    ie_dir = new_inv._fileid_to_entry_cache[b'dirid']
    self.assertEqual(['chïld'], sorted(ie_dir._children.keys()))