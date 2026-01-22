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
def jpeg_factory(fp=None, filename=None):
    im = JpegImageFile(fp, filename)
    try:
        mpheader = im._getmp()
        if mpheader[45057] > 1:
            from .MpoImagePlugin import MpoImageFile
            im = MpoImageFile.adopt(im, mpheader)
    except (TypeError, IndexError):
        pass
    except SyntaxError:
        warnings.warn('Image appears to be a malformed MPO file, it will be interpreted as a base JPEG file')
    return im