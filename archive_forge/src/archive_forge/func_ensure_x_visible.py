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
def ensure_x_visible(self, x: int) -> None:
    """Adjust `view_x` so that the given X coordinate is visible.

        The X coordinate is given relative to the current `view_x`.

        :Parameters:
            `x` : int
                X coordinate

        """
    x -= self.left
    if x <= self.view_x:
        self.view_x = x
    elif x >= self.view_x + self.width:
        self.view_x = x - self.width
    elif x >= self.view_x + self.width and self._content_width > self.width:
        self.view_x = x - self.width
    elif self.view_x + self.width > self._content_width:
        self.view_x = self._content_width