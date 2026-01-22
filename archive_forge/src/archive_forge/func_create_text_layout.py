import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def create_text_layout(self, text):
    text_buffer = create_unicode_buffer(text)
    text_layout = IDWriteTextLayout()
    hr = self._write_factory.CreateTextLayout(text_buffer, len(text_buffer), self._text_format, 10000, 80, byref(text_layout))
    return text_layout