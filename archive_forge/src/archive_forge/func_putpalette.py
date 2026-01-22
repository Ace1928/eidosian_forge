from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def putpalette(self, data, rawmode='RGB'):
    """
        Attaches a palette to this image.  The image must be a "P", "PA", "L"
        or "LA" image.

        The palette sequence must contain at most 256 colors, made up of one
        integer value for each channel in the raw mode.
        For example, if the raw mode is "RGB", then it can contain at most 768
        values, made up of red, green and blue values for the corresponding pixel
        index in the 256 colors.
        If the raw mode is "RGBA", then it can contain at most 1024 values,
        containing red, green, blue and alpha values.

        Alternatively, an 8-bit string may be used instead of an integer sequence.

        :param data: A palette sequence (either a list or a string).
        :param rawmode: The raw mode of the palette. Either "RGB", "RGBA", or a mode
           that can be transformed to "RGB" or "RGBA" (e.g. "R", "BGR;15", "RGBA;L").
        """
    from . import ImagePalette
    if self.mode not in ('L', 'LA', 'P', 'PA'):
        msg = 'illegal image mode'
        raise ValueError(msg)
    if isinstance(data, ImagePalette.ImagePalette):
        palette = ImagePalette.raw(data.rawmode, data.palette)
    else:
        if not isinstance(data, bytes):
            data = bytes(data)
        palette = ImagePalette.raw(rawmode, data)
    self._mode = 'PA' if 'A' in self.mode else 'P'
    self.palette = palette
    self.palette.mode = 'RGB'
    self.load()