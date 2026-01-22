import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_discard_one_parent(self):
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    root_entry_direntry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat), (b'd', b'', 0, False, packed_stat)])
    dirblocks = []
    dirblocks.append((b'', [root_entry_direntry]))
    dirblocks.append((b'', []))
    state = self.create_empty_dirstate()
    self.addCleanup(state.unlock)
    state._set_data([b'parent-id'], dirblocks[:])
    state._validate()
    state._discard_merge_parents()
    state._validate()
    self.assertEqual(dirblocks, state._dirblocks)