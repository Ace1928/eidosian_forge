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
def on_file_drop(self, x, y, paths):
    """File(s) were dropped into the window, will return the position of the cursor and
            a list of paths to the files that were dropped.

            .. versionadded:: 1.5.1

            :event:
            """