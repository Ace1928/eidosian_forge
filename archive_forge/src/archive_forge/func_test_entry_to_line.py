import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_entry_to_line(self):
    state = self.create_dirstate_with_root()
    try:
        self.assertEqual(b'\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk', state._entry_to_line(state._dirblocks[0][1][0]))
    finally:
        state.unlock()