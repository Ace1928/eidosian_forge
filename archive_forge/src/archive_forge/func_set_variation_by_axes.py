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
def set_variation_by_axes(self, axes):
    """
        :param axes: A list of values for each axis.
        :exception OSError: If the font is not a variation font.
        """
    try:
        self.font.setvaraxes(axes)
    except AttributeError as e:
        msg = 'FreeType 2.9.1 or greater is required'
        raise NotImplementedError(msg) from e