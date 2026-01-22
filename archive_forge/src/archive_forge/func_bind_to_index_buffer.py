import sys
import ctypes
from functools import lru_cache
import pyglet
from pyglet.gl import *
def bind_to_index_buffer(self):
    """Binds this buffer as an index buffer on the active vertex array."""
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.id)