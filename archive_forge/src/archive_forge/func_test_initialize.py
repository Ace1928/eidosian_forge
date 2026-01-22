import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_initialize(self):
    expected_result = ([], [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
    state = dirstate.DirState.initialize('dirstate')
    try:
        self.assertIsInstance(state, dirstate.DirState)
        lines = state.get_lines()
    finally:
        state.unlock()
    self.assertFileEqual(b''.join(lines), 'dirstate')
    state.lock_read()
    self.check_state_with_reopen(expected_result, state)