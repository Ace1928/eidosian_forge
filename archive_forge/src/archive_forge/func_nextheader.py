from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def nextheader(fobj):
    return struct.unpack('>4sI', fobj.read(HEADERSIZE))