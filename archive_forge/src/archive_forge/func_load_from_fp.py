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
def load_from_fp(self, fp, offset=None):
    self._loaded_exif = None
    self._data.clear()
    self._hidden_data.clear()
    self._ifds.clear()
    from . import TiffImagePlugin
    self.fp = fp
    if offset is not None:
        self.head = self._get_head()
    else:
        self.head = self.fp.read(8)
    self._info = TiffImagePlugin.ImageFileDirectory_v2(self.head)
    if self.endian is None:
        self.endian = self._info._endian
    if offset is None:
        offset = self._info.next
    self.fp.tell()
    self.fp.seek(offset)
    self._info.load(self.fp)