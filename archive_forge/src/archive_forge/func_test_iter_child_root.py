import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_iter_child_root(self):
    state, dirblocks = self.create_dirstate_with_two_trees()
    self.addCleanup(state.unlock)
    expected_result = []
    expected_result.append(dirblocks[1][1][0])
    expected_result.append(dirblocks[1][1][1])
    expected_result.append(dirblocks[1][1][3])
    expected_result.append(dirblocks[2][1][0])
    expected_result.append(dirblocks[2][1][1])
    expected_result.append(dirblocks[3][1][2])
    expected_result.append(dirblocks[3][1][3])
    expected_result.append(dirblocks[3][1][4])
    self.assertEqual(expected_result, list(state._iter_child_entries(1, b'')))