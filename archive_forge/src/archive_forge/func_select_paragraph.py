from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def select_paragraph(self, x: int, y: int) -> None:
    """Select the paragraph at the given window coordinate.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
    line = self._layout.get_line_from_point(x, y)
    p = self._layout.get_position_on_line(line, x)
    self.mark = self._layout.document.get_paragraph_start(p)
    self._position = self._layout.document.get_paragraph_end(p)
    self._update(line=line)
    self._next_attributes.clear()