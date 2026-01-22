import builtins
import sys
from ...utils.imports import _is_package_available
from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP
def write_choice(self, index, end: str=''):
    if sys.platform != 'win32':
        writeColor(self.choices[index], 32, end)
    else:
        forceWrite(self.choices[index], end)