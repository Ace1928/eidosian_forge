import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_path_to_root_no_parents_all_data(self):
    self.build_tree(['a file'])
    stat = os.lstat('a file')
    state = dirstate.DirState.initialize('dirstate')
    expected_entries = [((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', b'a file', b'a-file-id'), [(b'f', b'1' * 20, 19, False, dirstate.pack_stat(stat))])]
    try:
        state.add('a file', b'a-file-id', 'file', stat, b'1' * 20)
        self.assertEqual(expected_entries, list(state._iter_entries()))
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    self.addCleanup(state.unlock)
    self.assertEqual(expected_entries, list(state._iter_entries()))