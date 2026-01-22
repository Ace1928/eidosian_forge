import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_lt_by_dirs(self):
    if compiled_dirstate_helpers_feature.available():
        from .._dirstate_helpers_pyx import lt_by_dirs
    else:
        from .._dirstate_helpers_py import lt_by_dirs
    self.assertIs(lt_by_dirs, dirstate.lt_by_dirs)