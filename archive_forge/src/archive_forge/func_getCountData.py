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
def getCountData(self):
    v = self.table[self.name]
    if v is None:
        v = 0
    return {1: packUInt8, 2: packUShort, 4: packULong}[self.size](v)