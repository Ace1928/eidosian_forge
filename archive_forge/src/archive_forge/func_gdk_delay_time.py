from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
@property
def gdk_delay_time(self):
    assert self._iter is not None
    return gdkpixbuf.gdk_pixbuf_animation_iter_get_delay_time(self._iter)