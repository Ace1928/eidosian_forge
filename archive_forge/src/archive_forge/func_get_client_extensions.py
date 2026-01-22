from ctypes import *
from pyglet.gl.glx import *
from pyglet.util import asstr
def get_client_extensions(self):
    self.check_display()
    return asstr(glXGetClientString(self.display, GLX_EXTENSIONS)).split()