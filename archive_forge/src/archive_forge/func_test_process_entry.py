import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_process_entry(self):
    if compiled_dirstate_helpers_feature.available():
        from .._dirstate_helpers_pyx import ProcessEntryC
        self.assertIs(ProcessEntryC, dirstate._process_entry)
    else:
        from ..dirstate import ProcessEntryPython
        self.assertIs(ProcessEntryPython, dirstate._process_entry)