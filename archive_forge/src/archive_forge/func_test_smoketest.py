import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_smoketest(self):
    """Make sure that we can create and read back a simple file."""
    tree, state, expected = self.create_basic_dirstate()
    del tree
    state._read_header_if_needed()
    self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
    read_dirblocks = self.get_read_dirblocks()
    read_dirblocks(state)
    self.assertEqual(dirstate.DirState.IN_MEMORY_UNMODIFIED, state._dirblock_state)