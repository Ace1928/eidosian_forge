from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def isInt(f):
    try:
        i = int(f)
        if f - i == 0:
            return 1
        else:
            return 0
    except (ValueError, OverflowError):
        return 0