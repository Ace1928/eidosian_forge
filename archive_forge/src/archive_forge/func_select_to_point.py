from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def select_to_point(self, x: int, y: int) -> None:
    """Move the caret close to the given window coordinate while
        maintaining the `mark`.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
    line = self._layout.get_line_from_point(x, y)
    self._position = self._layout.get_position_on_line(line, x)
    self._update(line=line)
    self._next_attributes.clear()