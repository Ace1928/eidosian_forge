from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def readUInt24(self):
    pos = self.pos
    newpos = pos + 3
    value, = struct.unpack('>l', b'\x00' + self.data[pos:newpos])
    self.pos = newpos
    return value