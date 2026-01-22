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
@staticmethod
def unpack_localized_string(local_string: IDWriteLocalizedStrings, locale: str) -> List[str]:
    """Takes IDWriteLocalizedStrings and unpacks the strings inside of it into a list."""
    str_array_len = local_string.GetCount()
    strings = []
    for _ in range(str_array_len):
        string_size = UINT32()
        idx = Win32DirectWriteFont.get_localized_index(local_string, locale)
        local_string.GetStringLength(idx, byref(string_size))
        buffer_size = string_size.value
        buffer = create_unicode_buffer(buffer_size + 1)
        local_string.GetString(idx, buffer, len(buffer))
        strings.append(buffer.value)
    local_string.Release()
    return strings