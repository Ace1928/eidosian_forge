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
def load_memory(self, data, ext, filename='__inline__'):
    """(internal) Method to load an image from raw data.
        """
    self._filename = filename
    loaders = [loader for loader in ImageLoader.loaders if loader.can_load_memory() and ext in loader.extensions()]
    if not loaders:
        raise Exception('No inline loader found to load {}'.format(ext))
    image = loaders[0](filename, ext=ext, rawdata=data, inline=True, nocache=self._nocache, mipmap=self._mipmap, keep_data=self._keep_data)
    if isinstance(image, Texture):
        self._texture = image
        self._size = image.size
    else:
        self.image = image