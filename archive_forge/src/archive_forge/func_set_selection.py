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
def set_selection(self, start: int, end: int) -> None:
    """Set the text selection range.

        If ``start`` equals ``end`` no selection will be visible.

        :Parameters:
            `start` : int
                Starting character position of selection.
            `end` : int
                End of selection, exclusive.

        """
    start = max(0, start)
    end = min(end, len(self.document.text))
    if start == self._selection_start and end == self._selection_end:
        return
    if end > self._selection_start and start < self._selection_end:
        self.invalid_style.invalidate(min(start, self._selection_start), max(start, self._selection_start))
        self.invalid_style.invalidate(min(end, self._selection_end), max(end, self._selection_end))
    else:
        self.invalid_style.invalidate(self._selection_start, self._selection_end)
        self.invalid_style.invalidate(start, end)
    self._selection_start = start
    self._selection_end = end
    self._update()