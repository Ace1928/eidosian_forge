from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
def get_client_version(self):
    self.check_display()
    return asstr(glXGetClientString(self.display, GLX_VERSION))