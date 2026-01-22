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
def pack255UShort(value):
    """Encode unsigned integer in range 0 to 65535 (inclusive) to a bytestring
    using 255UInt16 variable-length encoding.

    >>> pack255UShort(252) == b'\\xfc'
    True
    >>> pack255UShort(506) == b'\\xfe\\x00'
    True
    >>> pack255UShort(762) == b'\\xfd\\x02\\xfa'
    True
    """
    if value < 0 or value > 65535:
        raise TTLibError('255UInt16 format requires 0 <= integer <= 65535')
    if value < 253:
        return struct.pack('>B', value)
    elif value < 506:
        return struct.pack('>BB', 255, value - 253)
    elif value < 762:
        return struct.pack('>BB', 254, value - 506)
    else:
        return struct.pack('>BH', 253, value)