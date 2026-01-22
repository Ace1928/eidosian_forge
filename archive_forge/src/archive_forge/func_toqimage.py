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
def toqimage(self):
    """Returns a QImage copy of this image"""
    from . import ImageQt
    if not ImageQt.qt_is_installed:
        msg = 'Qt bindings are not installed'
        raise ImportError(msg)
    return ImageQt.toqimage(self)