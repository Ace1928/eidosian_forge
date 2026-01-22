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
def register_mime(id, mimetype):
    """
    Registers an image MIME type by populating ``Image.MIME``. This function
    should not be used in application code.

    ``Image.MIME`` provides a mapping from image format identifiers to mime
    formats, but :py:meth:`~PIL.ImageFile.ImageFile.get_format_mimetype` can
    provide a different result for specific images.

    :param id: An image format identifier.
    :param mimetype: The image MIME type for this format.
    """
    MIME[id.upper()] = mimetype