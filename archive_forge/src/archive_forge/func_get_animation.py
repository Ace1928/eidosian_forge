from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def get_animation(self):
    self._finish_load()
    anim = gdkpixbuf.gdk_pixbuf_loader_get_animation(self._loader)
    if anim is None:
        raise ImageDecodeException('Failed to get animation from loader')
    gif_delays = self._get_gif_delays()
    return GdkPixBufAnimation(self, anim, gif_delays)