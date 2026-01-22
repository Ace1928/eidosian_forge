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
def point(self, lut, mode=None):
    """
        Maps this image through a lookup table or function.

        :param lut: A lookup table, containing 256 (or 65536 if
           self.mode=="I" and mode == "L") values per band in the
           image.  A function can be used instead, it should take a
           single argument. The function is called once for each
           possible pixel value, and the resulting table is applied to
           all bands of the image.

           It may also be an :py:class:`~PIL.Image.ImagePointHandler`
           object::

               class Example(Image.ImagePointHandler):
                 def point(self, data):
                   # Return result
        :param mode: Output mode (default is same as input).  In the
           current version, this can only be used if the source image
           has mode "L" or "P", and the output has mode "1" or the
           source image mode is "I" and the output mode is "L".
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
    self.load()
    if isinstance(lut, ImagePointHandler):
        return lut.point(self)
    if callable(lut):
        if self.mode in ('I', 'I;16', 'F'):
            scale, offset = _getscaleoffset(lut)
            return self._new(self.im.point_transform(scale, offset))
        lut = [lut(i) for i in range(256)] * self.im.bands
    if self.mode == 'F':
        msg = 'point operation not supported for this mode'
        raise ValueError(msg)
    if mode != 'F':
        lut = [round(i) for i in lut]
    return self._new(self.im.point(lut, mode))