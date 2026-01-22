import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def set_breakpoint(self, line=None, symbol=None, condition=None):
    """gdb command ~= 'break'"""
    if line is not None and symbol is not None:
        raise ValueError('Can only supply one of line or symbol')
    bp = '-break-insert '
    if condition is not None:
        bp += f'-c "{condition}" '
    if line is not None:
        assert isinstance(line, int)
        bp += f'-f {self._file_name}:{line} '
    if symbol is not None:
        assert isinstance(symbol, str)
        bp += f'-f {symbol} '
    self._run_command(bp, expect='\\^done')