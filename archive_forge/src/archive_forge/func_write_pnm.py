from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def write_pnm(file, width, height, pixels, meta):
    """Write a Netpbm PNM/PAM file."""
    bitdepth = meta['bitdepth']
    maxval = 2 ** bitdepth - 1
    planes = meta['planes']
    assert planes in (1, 2, 3, 4)
    if planes in (1, 3):
        if 1 == planes:
            fmt = 'P5'
        else:
            fmt = 'P6'
        file.write('%s %d %d %d\n' % (fmt, width, height, maxval))
    if planes in (2, 4):
        if 2 == planes:
            tupltype = 'GRAYSCALE_ALPHA'
        else:
            tupltype = 'RGB_ALPHA'
        file.write('P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n' % (width, height, planes, maxval, tupltype))
    vpr = planes * width
    fmt = '>%d' % vpr
    if maxval > 255:
        fmt = fmt + 'H'
    else:
        fmt = fmt + 'B'
    for row in pixels:
        file.write(struct.pack(fmt, *row))
    file.flush()