import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_entry_to_line_with_parent(self):
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    root_entry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat), (b'a', b'dirname/basename', 0, False, b'')])
    state = dirstate.DirState.initialize('dirstate')
    try:
        self.assertEqual(b'\x00\x00a-root-value\x00d\x00\x000\x00n\x00AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk\x00a\x00dirname/basename\x000\x00n\x00', state._entry_to_line(root_entry))
    finally:
        state.unlock()