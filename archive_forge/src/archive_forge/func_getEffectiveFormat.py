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
def getEffectiveFormat(self):
    format = 0
    for name, value in self.__dict__.items():
        if value:
            format = format | valueRecordFormatDict[name][0]
    return format