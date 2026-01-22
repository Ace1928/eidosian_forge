from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def read_sis(fh: FileHandle, byteorder: ByteOrder, dtype: int, count: int, offsetsize: int, /) -> dict[str, Any]:
    """Read OlympusSIS structure from file.

    No specification is available. Only few fields are known.

    """
    result: dict[str, Any] = {}
    magic, minute, hour, day, month, year, name, tagcount = struct.unpack('<4s6xhhhhh6x32sh', fh.read(60))
    if magic != b'SIS0':
        raise ValueError('invalid OlympusSIS structure')
    result['name'] = bytes2str(stripnull(name))
    try:
        result['datetime'] = datetime.datetime(1900 + year, month + 1, day, hour, minute)
    except ValueError:
        pass
    data = fh.read(8 * tagcount)
    for i in range(0, tagcount * 8, 8):
        tagtype, count, offset = struct.unpack('<hhI', data[i:i + 8])
        fh.seek(offset)
        if tagtype == 1:
            lenexp, xcal, ycal, mag, camname, pictype = struct.unpack('<10xhdd8xd2x34s32s', fh.read(112))
            m = math.pow(10, lenexp)
            result['pixelsizex'] = xcal * m
            result['pixelsizey'] = ycal * m
            result['magnification'] = mag
            result['cameraname'] = bytes2str(stripnull(camname))
            result['picturetype'] = bytes2str(stripnull(pictype))
        elif tagtype == 10:
            continue
    return result