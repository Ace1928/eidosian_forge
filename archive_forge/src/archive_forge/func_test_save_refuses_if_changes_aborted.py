import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_save_refuses_if_changes_aborted(self):
    self.build_tree(['a-file', 'a-dir/'])
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.add('a-file', b'a-file-id', 'file', None, b'')
        state.save()
    finally:
        state.unlock()
    expected_blocks = [(b'', [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])]), (b'', [((b'', b'a-file', b'a-file-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT)])])]
    state = dirstate.DirState.on_file('dirstate')
    state.lock_write()
    try:
        state._read_dirblocks_if_needed()
        self.assertEqual(expected_blocks, state._dirblocks)
        state.add('a-dir', b'a-dir-id', 'directory', None, b'')
        state._changes_aborted = True
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        state._read_dirblocks_if_needed()
        self.assertEqual(expected_blocks, state._dirblocks)
    finally:
        state.unlock()