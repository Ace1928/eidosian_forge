import platform
from ctypes import c_uint32, c_int, byref
from pyglet.gl.base import Config, CanvasConfig, Context
from pyglet.gl import ContextException
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, quartz
def os_x_version():
    version = tuple([int(v) for v in platform.release().split('.')])
    if len(version) > 0:
        return version
    return (version,)