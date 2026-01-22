from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def load_djpeg(self):
    f, path = tempfile.mkstemp()
    os.close(f)
    if os.path.exists(self.filename):
        subprocess.check_call(['djpeg', '-outfile', path, self.filename])
    else:
        try:
            os.unlink(path)
        except OSError:
            pass
        msg = 'Invalid Filename'
        raise ValueError(msg)
    try:
        with Image.open(path) as _im:
            _im.load()
            self.im = _im.im
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    self._mode = self.im.mode
    self._size = self.im.size
    self.tile = []