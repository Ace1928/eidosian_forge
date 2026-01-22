import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def unset_state_recursive(self):
    """Unset this group and its ancestry.

        The inverse of `set_state_recursive`.
        """
    self.unset_state()
    if self.parent:
        self.parent.unset_state_recursive()