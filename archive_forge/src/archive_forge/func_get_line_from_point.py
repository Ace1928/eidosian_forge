from __future__ import annotations
import sys
from typing import List, Optional, Type, Any, Tuple, TYPE_CHECKING
from pyglet.customtypes import AnchorX, AnchorY
from pyglet.event import EventDispatcher
from pyglet.font.base import grapheme_break
from pyglet.text import runlist
from pyglet.text.document import AbstractDocument
from pyglet.text.layout.base import _is_pyglet_doc_run, _Line, _LayoutContext, _InlineElementBox, _InvalidRange, \
from pyglet.text.layout.scrolling import ScrollableTextLayoutGroup, ScrollableTextDecorationGroup
def get_line_from_point(self, x, y):
    """Get the closest line index to a point.

        :Parameters:
            `x` : int
                X coordinate.
            `y` : int
                Y coordinate.

        :rtype: int
        """
    x -= self._translate_x
    y -= self._get_content_height() + self.bottom - self._translate_y
    line_index = 0
    for line in self.lines:
        if y > line.y + line.descent:
            break
        line_index += 1
    if line_index >= len(self.lines):
        line_index = len(self.lines) - 1
    return line_index