from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
def push_style(self, key, styles):
    old_styles = {}
    for name in styles.keys():
        old_styles[name] = self.current_style.get(name)
    self.stack.append((key, old_styles))
    self.current_style.update(styles)
    self.next_style.update(styles)