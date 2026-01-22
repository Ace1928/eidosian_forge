import binascii
import math
import struct
import sys
import pytest
from shapely import wkt
from shapely.geometry import Point
from shapely.geos import geos_version
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads
def hostorder(fmt, value):
    """Re-pack a hex WKB value to native endianness if needed

    This routine does not understand WKB format, so it must be provided a
    struct module format string, without initial indicator character ("@=<>!"),
    which will be interpreted as big- or little-endian with standard sizes
    depending on the endian flag in the first byte of the value.
    """
    if fmt and fmt[0] in '@=<>!':
        raise ValueError('Initial indicator character, one of @=<>!, in fmt')
    if not fmt or fmt[0] not in 'cbB':
        raise ValueError('Missing endian flag in fmt')
    hexendian, = struct.unpack(fmt[0], hex2bin(value[:2]))
    hexorder = {0: '>', 1: '<'}[hexendian]
    sysorder = {'little': '<', 'big': '>'}[sys.byteorder]
    if hexorder == sysorder:
        return value
    return bin2hex(struct.pack(sysorder + fmt, {'>': 0, '<': 1}[sysorder], *struct.unpack(hexorder + fmt, hex2bin(value))[1:]))