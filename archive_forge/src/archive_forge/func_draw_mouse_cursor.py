import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
def draw_mouse_cursor(self):
    """Draw the custom mouse cursor.

        If the current mouse cursor has ``drawable`` set, this method
        is called before the buffers are flipped to render it.

        There is little need to override this method; instead, subclass
        :py:class:`MouseCursor` and provide your own
        :py:meth:`~MouseCursor.draw` method.
        """
    if self._mouse_cursor.gl_drawable and self._mouse_visible and self._mouse_in_window:
        self._mouse_cursor.draw(self._mouse_x, self._mouse_y)