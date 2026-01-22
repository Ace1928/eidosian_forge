import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_state_from_scratch_identical_parent(self):
    tree1, revid1 = self.make_minimal_tree()
    inv = tree1.root_inventory
    root_id = inv.path2id('')
    rev_tree1 = tree1.branch.repository.revision_tree(revid1)
    d_entry = (b'd', b'', 0, False, dirstate.DirState.NULLSTAT)
    parent_entry = (b'd', b'', 0, False, revid1)
    expected_result = ([revid1], [((b'', b'', root_id), [d_entry, parent_entry])])
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_state_from_scratch(inv, [(revid1, rev_tree1)], [])
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._header_state)
        self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
    except:
        state.unlock()
        raise
    else:
        self.check_state_with_reopen(expected_result, state)