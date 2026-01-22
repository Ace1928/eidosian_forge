import platform
from ctypes import c_uint32, c_int, byref
from pyglet.gl.base import Config, CanvasConfig, Context
from pyglet.gl import ContextException
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, quartz
def get_vsync(self):
    vals = c_int()
    self._nscontext.getValues_forParameter_(byref(vals), cocoapy.NSOpenGLCPSwapInterval)
    return vals.value