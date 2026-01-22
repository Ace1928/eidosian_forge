from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
def update_anchor(self, anchor_x: float, anchor_y: float) -> None:
    anchor = (anchor_x, anchor_y)
    for _vertex_list in self.vertex_lists.values():
        _vertex_list.anchor[:] = anchor * _vertex_list.count