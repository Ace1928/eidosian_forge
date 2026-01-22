import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_cmp_non_ascii(self):
    self.assertCmpByDirs(-1, b'\xc2\xb5', b'\xc3\xa5')
    self.assertCmpByDirs(-1, b'a', b'\xc3\xa5')
    self.assertCmpByDirs(-1, b'b', b'\xc2\xb5')
    self.assertCmpByDirs(-1, b'a/b', b'a/\xc3\xa5')
    self.assertCmpByDirs(-1, b'b/a', b'b/\xc2\xb5')