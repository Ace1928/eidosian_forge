import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_can_save_in_read_lock(self):
    state = self.create_updated_dirstate()
    try:
        entry = state._get_entry(0, path_utf8=b'a-file')
        self.assertEqual(0, entry[1][0][2])
        self.assertNotEqual((None, None), entry)
        state._sha_cutoff_time()
        state._cutoff_time += 10.0
        st = os.lstat('a-file')
        sha1sum = dirstate.update_entry(state, entry, 'a-file', st)
        self.assertEqual(b'ecc5374e9ed82ad3ea3b4d452ea995a5fd3e70e3', sha1sum)
        self.assertEqual(st.st_size, entry[1][0][2])
        self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
        del entry
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        entry = state._get_entry(0, path_utf8=b'a-file')
        self.assertEqual(st.st_size, entry[1][0][2])
    finally:
        state.unlock()