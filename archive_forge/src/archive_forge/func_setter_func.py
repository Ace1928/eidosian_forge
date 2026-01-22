import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def setter_func(values):
    glUseProgram(program)
    c_array[:] = values
    gl_setter(location, 1, ptr)