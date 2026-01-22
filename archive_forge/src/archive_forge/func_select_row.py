import builtins
import sys
from ...utils.imports import _is_package_available
from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP
@input.mark_multiple(*[KEYMAP[str(number)] for number in range(10)])
def select_row(self):
    index = int(chr(self.current_selection))
    movement = index - self.position
    if index == self.position:
        return
    if index < len(self.choices):
        if self.position > index:
            self.move_direction(Direction.UP, -movement)
        elif self.position < index:
            self.move_direction(Direction.DOWN, movement)
        else:
            return
    else:
        return