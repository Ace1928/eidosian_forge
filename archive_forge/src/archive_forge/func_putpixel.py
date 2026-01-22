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
def putpixel(self, xy, value):
    """
        Modifies the pixel at the given position. The color is given as
        a single numerical value for single-band images, and a tuple for
        multi-band images. In addition to this, RGB and RGBA tuples are
        accepted for P and PA images.

        Note that this method is relatively slow.  For more extensive changes,
        use :py:meth:`~PIL.Image.Image.paste` or the :py:mod:`~PIL.ImageDraw`
        module instead.

        See:

        * :py:meth:`~PIL.Image.Image.paste`
        * :py:meth:`~PIL.Image.Image.putdata`
        * :py:mod:`~PIL.ImageDraw`

        :param xy: The pixel coordinate, given as (x, y). See
           :ref:`coordinate-system`.
        :param value: The pixel value.
        """
    if self.readonly:
        self._copy()
    self.load()
    if self.pyaccess:
        return self.pyaccess.putpixel(xy, value)
    if self.mode in ('P', 'PA') and isinstance(value, (list, tuple)) and (len(value) in [3, 4]):
        if self.mode == 'PA':
            alpha = value[3] if len(value) == 4 else 255
            value = value[:3]
        value = self.palette.getcolor(value, self)
        if self.mode == 'PA':
            value = (value, alpha)
    return self.im.putpixel(xy, value)