import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@property
def uniform_blocks(self) -> dict:
    return self._uniform_blocks