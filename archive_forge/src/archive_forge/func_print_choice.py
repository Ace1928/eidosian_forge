import builtins
import sys
from ...utils.imports import _is_package_available
from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP
def print_choice(self, index: int):
    """Prints the choice at the given index"""
    if index == self.position:
        forceWrite(f' {self.arrow_char} ')
        self.write_choice(index)
    else:
        forceWrite(f'    {self.choices[index]}')
    reset_cursor()