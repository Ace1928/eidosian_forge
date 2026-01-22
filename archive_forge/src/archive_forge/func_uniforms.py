import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
@property
def uniforms(self):
    return {n: dict(location=u.location, length=u.length, size=u.size) for n, u in self._uniforms.items()}