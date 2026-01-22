from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def get_pixbuf(self):
    self._finish_load()
    pixbuf = gdkpixbuf.gdk_pixbuf_loader_get_pixbuf(self._loader)
    if pixbuf is None:
        raise ImageDecodeException('Failed to get pixbuf from loader')
    return GdkPixBuf(self, pixbuf)