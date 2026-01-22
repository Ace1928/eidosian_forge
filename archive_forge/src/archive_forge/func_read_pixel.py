import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
def read_pixel(self, x, y):
    """For a given local x/y position, return the pixel color at that
        position.

        .. warning::
            This function can only be used with images loaded with the
            keep_data=True keyword. For example::

                m = Image.load('image.png', keep_data=True)
                color = m.read_pixel(150, 150)

        :Parameters:
            `x`: int
                Local x coordinate of the pixel in question.
            `y`: int
                Local y coordinate of the pixel in question.
        """
    data = self.image._data[0]
    if data.data is None:
        raise EOFError('Image data is missing, make sure that image isloaded with keep_data=True keyword.')
    x, y = (int(x), int(y))
    if not (0 <= x < data.width and 0 <= y < data.height):
        raise IndexError('Position (%d, %d) is out of range.' % (x, y))
    assert data.fmt in ImageData._supported_fmts
    size = 3 if data.fmt in ('rgb', 'bgr') else 4
    index = y * data.width * size + x * size
    raw = bytearray(data.data[index:index + size])
    color = [c / 255.0 for c in raw]
    bgr_flag = False
    if data.fmt == 'argb':
        color.reverse()
        bgr_flag = True
    elif data.fmt == 'abgr':
        color.reverse()
    if bgr_flag or data.fmt in ('bgr', 'bgra'):
        color[0], color[2] = (color[2], color[0])
    return color