from __future__ import annotations
from typing import Type, Optional, TYPE_CHECKING, Tuple
from pyglet import graphics
from pyglet.customtypes import AnchorY, AnchorX
from pyglet.gl import glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
from pyglet.text.layout.base import TextLayout
Vertical scroll offset.

            The initial value is 0, and the top of the text will touch the top of the
            layout bounds (unless the content height is less than the layout height,
            in which case `content_valign` is used).

            A negative value causes the text to "scroll" upwards.  Values outside of
            the range ``[height - content_height, 0]`` are automatically clipped in
            range.

            :type: int
        