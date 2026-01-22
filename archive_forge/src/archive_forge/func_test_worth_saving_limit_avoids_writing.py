import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_worth_saving_limit_avoids_writing(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['c', 'd'])
    tree.lock_write()
    tree.add(['c', 'd'], ids=[b'c-id', b'd-id'])
    tree.commit('add c and d')
    state = InstrumentedDirState.on_file(tree.current_dirstate()._filename, worth_saving_limit=2)
    tree.unlock()
    state.lock_write()
    self.addCleanup(state.unlock)
    state._read_dirblocks_if_needed()
    state.adjust_time(+20)
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    content = self._read_state_content(state)
    self.do_update_entry(state, b'c')
    self.assertEqual(1, len(state._known_hash_changes))
    self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
    self.assertEqual(content, self._read_state_content(state))
    self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)
    self.do_update_entry(state, b'd')
    self.assertEqual(2, len(state._known_hash_changes))
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    self.assertEqual(0, len(state._known_hash_changes))