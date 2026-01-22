from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_some_files(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'], ids=[b'thefileid'])
    revid = wt.commit('commit')
    tree = wt.branch.repository.revision_tree(revid)
    inv = mutable_inventory_from_tree(tree)
    self.assertEqual(revid, inv.revision_id)
    self.assertEqual(2, len(inv))
    self.assertEqual('a', inv.get_entry(b'thefileid').name)
    self.assertFalse(tree.root_inventory.get_entry(b'thefileid').executable)
    inv.get_entry(b'thefileid').executable = True
    self.assertFalse(tree.root_inventory.get_entry(b'thefileid').executable)