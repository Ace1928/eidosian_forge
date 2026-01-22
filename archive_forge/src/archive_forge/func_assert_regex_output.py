import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def assert_regex_output(self, expected):
    """Asserts that the current output string contains the expected
        regex."""
    output = self._captured.after
    decoded = output.decode('utf-8')
    done_str = decoded.splitlines()[0]
    found = re.match(expected, done_str)
    assert found, f'decoded={decoded}\nexpected={expected})'