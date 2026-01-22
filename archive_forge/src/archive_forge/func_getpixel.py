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
def getpixel(self, xy):
    """
        Returns the pixel value at a given position.

        :param xy: The coordinate, given as (x, y). See
           :ref:`coordinate-system`.
        :returns: The pixel value.  If the image is a multi-layer image,
           this method returns a tuple.
        """
    self.load()
    if self.pyaccess:
        return self.pyaccess.getpixel(xy)
    return self.im.getpixel(tuple(xy))