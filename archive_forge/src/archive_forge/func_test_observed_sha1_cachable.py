import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_observed_sha1_cachable(self):
    state, entry = self.get_state_with_a()
    state.save()
    atime = time.time() - 10
    self.build_tree(['a'])
    statvalue = test_dirstate._FakeStat.from_stat(os.lstat('a'))
    statvalue.st_mtime = statvalue.st_ctime = atime
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    state._observed_sha1(entry, b'foo', statvalue)
    self.assertEqual(b'foo', entry[1][0][1])
    packed_stat = dirstate.pack_stat(statvalue)
    self.assertEqual(packed_stat, entry[1][0][4])
    self.assertEqual(dirstate.DirState.IN_MEMORY_HASH_MODIFIED, state._dirblock_state)