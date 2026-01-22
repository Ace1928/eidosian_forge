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
def load_from_bytes(f):
    self.font_bytes = f.read()
    self.font = core.getfont('', size, index, encoding, self.font_bytes, layout_engine)