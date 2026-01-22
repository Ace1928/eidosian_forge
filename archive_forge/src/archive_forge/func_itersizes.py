from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def itersizes(self):
    sizes = []
    for size, fmts in self.SIZES.items():
        for fmt, reader in fmts:
            if fmt in self.dct:
                sizes.append(size)
                break
    return sizes