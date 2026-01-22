import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_observed_sha1_not_cachable(self):
    state, entry = self.get_state_with_a()
    state.save()
    oldval = entry[1][0][1]
    oldstat = entry[1][0][4]
    self.build_tree(['a'])
    statvalue = os.lstat('a')
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)
    state._observed_sha1(entry, 'foo', statvalue)
    self.assertEqual(oldval, entry[1][0][1])
    self.assertEqual(oldstat, entry[1][0][4])
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)