import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def stat_and_sha1(self, abspath):
    with open(abspath, 'rb') as file_obj:
        statvalue = os.fstat(file_obj.fileno())
        text = b''.join(file_obj.readlines())
        sha1 = osutils.sha_string(text.upper() + b'foo')
    return (statvalue, sha1)