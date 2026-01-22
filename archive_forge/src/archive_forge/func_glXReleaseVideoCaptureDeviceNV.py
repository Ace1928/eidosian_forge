from OpenGL import platform as _p, arrays
from OpenGL.raw.GLX import _types as _cs
from OpenGL.raw.GLX._types import *
from OpenGL.raw.GLX import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, ctypes.POINTER(_cs.Display), _cs.GLXVideoCaptureDeviceNV)
def glXReleaseVideoCaptureDeviceNV(dpy, device):
    pass