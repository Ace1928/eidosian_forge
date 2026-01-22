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
def getDataLength(self):
    """Return the length of this table in bytes, without subtables."""
    l = 0
    for item in self.items:
        if hasattr(item, 'getCountData'):
            l += item.size
        elif hasattr(item, 'subWriter'):
            l += item.offsetSize
        else:
            l = l + len(item)
    return l