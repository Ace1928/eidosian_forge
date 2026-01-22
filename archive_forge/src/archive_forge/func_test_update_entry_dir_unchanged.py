import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry_dir_unchanged(self):
    state, entry = self.get_state_with_a()
    self.build_tree(['a/'])
    state.adjust_time(+20)
    self.assertIs(None, self.do_update_entry(state, entry, b'a'))
    self.assertEqual(dirstate.DirState.IN_MEMORY_MODIFIED, state._dirblock_state)
    state.save()
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    self.assertIs(None, self.do_update_entry(state, entry, b'a'))
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    t = time.time() - 100.0
    try:
        os.utime('a', (t, t))
    except OSError:
        raise tests.TestSkipped("can't update mtime of a dir on FAT")
    saved_packed_stat = entry[1][0][-1]
    self.assertIs(None, self.do_update_entry(state, entry, b'a'))
    self.assertNotEqual(saved_packed_stat, entry[1][0][-1])
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)