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
def putdata(self, data, scale=1.0, offset=0.0):
    """
        Copies pixel data from a flattened sequence object into the image. The
        values should start at the upper left corner (0, 0), continue to the
        end of the line, followed directly by the first value of the second
        line, and so on. Data will be read until either the image or the
        sequence ends. The scale and offset values are used to adjust the
        sequence values: **pixel = value*scale + offset**.

        :param data: A flattened sequence object.
        :param scale: An optional scale value.  The default is 1.0.
        :param offset: An optional offset value.  The default is 0.0.
        """
    self._ensure_mutable()
    self.im.putdata(data, scale, offset)