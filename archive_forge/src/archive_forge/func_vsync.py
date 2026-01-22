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
@property
def vsync(self):
    """True if buffer flips are synchronised to the screen's vertical
        retrace.  Read-only.

        :type: bool
        """
    return self._vsync