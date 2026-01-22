import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_state_from_scratch_no_parents(self):
    tree1, revid1 = self.make_minimal_tree()
    inv = tree1.root_inventory
    root_id = inv.path2id('')
    expected_result = ([], [((b'', b'', root_id), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_state_from_scratch(inv, [], [])
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._header_state)
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
    except:
        state.unlock()
        raise
    else:
        self.check_state_with_reopen(expected_result, state)