from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
def get_variation_names(self):
    """
        :returns: A list of the named styles in a variation font.
        :exception OSError: If the font is not a variation font.
        """
    try:
        names = self.font.getvarnames()
    except AttributeError as e:
        msg = 'FreeType 2.9.1 or greater is required'
        raise NotImplementedError(msg) from e
    return [name.replace(b'\x00', b'') for name in names]