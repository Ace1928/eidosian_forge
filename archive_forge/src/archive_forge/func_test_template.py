import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest
def test_template(self):
    o, e = self.run_test_in_separate_process(injected_method)
    dbgmsg = f'\nSTDOUT={o}\nSTDERR={e}\n'
    m = re.search("\\.\\.\\. skipped '(.*?)'", e)
    if m is not None:
        self.skipTest(m.group(1))
    self.assertIn('GNU gdb', o, msg=dbgmsg)
    self.assertIn('OK', e, msg=dbgmsg)
    self.assertNotIn('FAIL', e, msg=dbgmsg)
    self.assertNotIn('ERROR', e, msg=dbgmsg)