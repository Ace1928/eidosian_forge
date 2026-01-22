from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
def pop_style(self, key):
    for match, _ in self.stack:
        if key == match:
            break
    else:
        return
    while True:
        match, old_styles = self.stack.pop()
        self.next_style.update(old_styles)
        self.current_style.update(old_styles)
        if match == key:
            break