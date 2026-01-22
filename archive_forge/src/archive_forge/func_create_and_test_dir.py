import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def create_and_test_dir(self, state, entry):
    """Create a directory at 'a' and verify the state finds it.

        The state should already be versioning *something* at 'a'. This makes
        sure that state.update_entry recognizes it as a directory.
        """
    self.build_tree(['a/'])
    stat_value = os.lstat('a')
    packed_stat = dirstate.pack_stat(stat_value)
    link_or_sha1 = self.do_update_entry(state, entry, abspath=b'a')
    self.assertIs(None, link_or_sha1)
    self.assertEqual([(b'd', b'', 0, False, packed_stat)], entry[1])
    return packed_stat