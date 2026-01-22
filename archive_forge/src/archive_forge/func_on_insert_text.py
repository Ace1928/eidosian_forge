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
def on_insert_text(self, start: int, text: str) -> None:
    len_text = len(text)
    self.glyphs[start:start] = [None] * len_text
    self.invalid_glyphs.insert(start, len_text)
    if self._multiline and self._content_valign != 'top':
        visible_line = self.lines[self.visible_lines.start]
        self.invalid_flow.invalidate(visible_line.start, start + len_text)
    self.invalid_flow.insert(start, len_text)
    self.invalid_style.insert(start, len_text)
    self.owner_runs.insert(start, len_text)
    for line in self.lines:
        if line.start >= start:
            line.start += len_text
    self._update()