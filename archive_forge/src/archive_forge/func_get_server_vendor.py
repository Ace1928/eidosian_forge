from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
def get_server_vendor(self):
    self.check_display()
    return asstr(glXQueryServerString(self.display, 0, GLX_VENDOR))