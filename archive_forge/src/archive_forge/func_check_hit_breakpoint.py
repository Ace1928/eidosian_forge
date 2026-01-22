import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def check_hit_breakpoint(self, number=None, line=None):
    """Checks that a breakpoint has been hit"""
    self._captured_expect('\\*stopped,.*\\r\\n')
    self.assert_output('*stopped,reason="breakpoint-hit",')
    if number is not None:
        assert isinstance(number, int)
        self.assert_output(f'bkptno="{number}"')
    if line is not None:
        assert isinstance(line, int)
        self.assert_output(f'line="{line}"')