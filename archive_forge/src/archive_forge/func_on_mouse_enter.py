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
def on_mouse_enter(self, x, y):
    """The mouse was moved into the window.

            This event will not be triggered if the mouse is currently being
            dragged.

            :Parameters:
                `x` : int
                    Distance in pixels from the left edge of the window.
                `y` : int
                    Distance in pixels from the bottom edge of the window.

            :event:
            """