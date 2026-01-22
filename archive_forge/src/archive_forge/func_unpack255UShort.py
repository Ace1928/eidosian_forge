from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def unpack255UShort(data):
    """Read one to three bytes from 255UInt16-encoded input string, and return a
    tuple containing the decoded integer plus any leftover data.

    >>> unpack255UShort(bytechr(252))[0]
    252

    Note that some numbers (e.g. 506) can have multiple encodings:
    >>> unpack255UShort(struct.pack("BB", 254, 0))[0]
    506
    >>> unpack255UShort(struct.pack("BB", 255, 253))[0]
    506
    >>> unpack255UShort(struct.pack("BBB", 253, 1, 250))[0]
    506
    """
    code = byteord(data[:1])
    data = data[1:]
    if code == 253:
        if len(data) < 2:
            raise TTLibError('not enough data to unpack 255UInt16')
        result, = struct.unpack('>H', data[:2])
        data = data[2:]
    elif code == 254:
        if len(data) == 0:
            raise TTLibError('not enough data to unpack 255UInt16')
        result = byteord(data[:1])
        result += 506
        data = data[1:]
    elif code == 255:
        if len(data) == 0:
            raise TTLibError('not enough data to unpack 255UInt16')
        result = byteord(data[:1])
        result += 253
        data = data[1:]
    else:
        result = code
    return (result, data)