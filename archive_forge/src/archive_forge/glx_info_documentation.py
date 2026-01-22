from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
Information about version and extensions of current GLX implementation.

Usage::

    from pyglet.gl import glx_info

    if glx_info.have_extension('GLX_NV_float_buffer'):
        # ...

Or, if using more than one display::

    from pyglet.gl.glx_info import GLXInfo

    info = GLXInfo(window._display)
    if info.get_server_vendor() == 'ATI':
        # ...

