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
def hide_offsets(self):
    for tag in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
        if tag in self:
            self._hidden_data[tag] = self[tag]
            del self[tag]