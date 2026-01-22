import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_sha1provider_sha1_used(self):
    tree, text = self._prepare_tree()
    state = dirstate.DirState.from_tree(tree, 'dirstate', UppercaseSHA1Provider())
    self.addCleanup(state.unlock)
    expected_sha = osutils.sha_string(text.upper() + b'foo')
    entry = state._get_entry(0, path_utf8=b'a file')
    self.assertNotEqual((None, None), entry)
    state._sha_cutoff_time()
    state._cutoff_time += 10
    sha1 = self.update_entry(state, entry, 'tree/a file', os.lstat('tree/a file'))
    self.assertEqual(expected_sha, sha1)