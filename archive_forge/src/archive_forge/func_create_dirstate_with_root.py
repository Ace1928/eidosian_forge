import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_dirstate_with_root(self):
    """Return a write-locked state with a single root entry."""
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    root_entry_direntry = ((b'', b'', b'a-root-value'), [(b'd', b'', 0, False, packed_stat)])
    dirblocks = []
    dirblocks.append((b'', [root_entry_direntry]))
    dirblocks.append((b'', []))
    state = self.create_empty_dirstate()
    try:
        state._set_data([], dirblocks)
        state._validate()
    except:
        state.unlock()
        raise
    return state