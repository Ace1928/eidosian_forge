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
def filter_scanline(type, line, fo, prev=None):
    """Apply a scanline filter to a scanline.  `type` specifies the
    filter type (0 to 4); `line` specifies the current (unfiltered)
    scanline as a sequence of bytes; `prev` specifies the previous
    (unfiltered) scanline as a sequence of bytes. `fo` specifies the
    filter offset; normally this is size of a pixel in bytes (the number
    of bytes per sample times the number of channels), but when this is
    < 1 (for bit depths < 8) then the filter offset is 1.
    """
    assert 0 <= type < 5
    out = array('B', [type])

    def sub():
        ai = -fo
        for x in line:
            if ai >= 0:
                x = x - line[ai] & 255
            out.append(x)
            ai += 1

    def up():
        for i, x in enumerate(line):
            x = x - prev[i] & 255
            out.append(x)

    def average():
        ai = -fo
        for i, x in enumerate(line):
            if ai >= 0:
                x = x - (line[ai] + prev[i] >> 1) & 255
            else:
                x = x - (prev[i] >> 1) & 255
            out.append(x)
            ai += 1

    def paeth():
        ai = -fo
        for i, x in enumerate(line):
            a = 0
            b = prev[i]
            c = 0
            if ai >= 0:
                a = line[ai]
                c = prev[ai]
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                Pr = a
            elif pb <= pc:
                Pr = b
            else:
                Pr = c
            x = x - Pr & 255
            out.append(x)
            ai += 1
    if not prev:
        if type == 2:
            return line
        elif type == 3:
            prev = [0] * len(line)
        elif type == 4:
            type = 1
    if type == 0:
        out.extend(line)
    elif type == 1:
        sub()
    elif type == 2:
        up()
    elif type == 3:
        average()
    else:
        paeth()
    return out