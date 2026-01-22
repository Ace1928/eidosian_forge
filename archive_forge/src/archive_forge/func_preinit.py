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
def preinit():
    """
    Explicitly loads BMP, GIF, JPEG, PPM and PPM file format drivers.

    It is called when opening or saving images.
    """
    global _initialized
    if _initialized >= 1:
        return
    try:
        from . import BmpImagePlugin
        assert BmpImagePlugin
    except ImportError:
        pass
    try:
        from . import GifImagePlugin
        assert GifImagePlugin
    except ImportError:
        pass
    try:
        from . import JpegImagePlugin
        assert JpegImagePlugin
    except ImportError:
        pass
    try:
        from . import PpmImagePlugin
        assert PpmImagePlugin
    except ImportError:
        pass
    try:
        from . import PngImagePlugin
        assert PngImagePlugin
    except ImportError:
        pass
    _initialized = 1