import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
@staticmethod
def unpack_field(packed_string, stat_field):
    return _dirstate_helpers_py._unpack_stat(packed_string)[stat_field]