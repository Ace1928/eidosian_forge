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
def deinterlace(self, raw):
    """
        Read raw pixel data, undo filters, deinterlace, and flatten.
        Return in flat row flat pixel format.
        """
    vpr = self.width * self.planes
    fmt = 'BH'[self.bitdepth > 8]
    a = array(fmt, [0] * vpr * self.height)
    source_offset = 0
    for xstart, ystart, xstep, ystep in _adam7:
        if xstart >= self.width:
            continue
        recon = None
        ppr = int(math.ceil((self.width - xstart) / float(xstep)))
        row_size = int(math.ceil(self.psize * ppr))
        for y in range(ystart, self.height, ystep):
            filter_type = raw[source_offset]
            source_offset += 1
            scanline = raw[source_offset:source_offset + row_size]
            source_offset += row_size
            recon = self.undo_filter(filter_type, scanline, recon)
            flat = self.serialtoflat(recon, ppr)
            if xstep == 1:
                assert xstart == 0
                offset = y * vpr
                a[offset:offset + vpr] = flat
            else:
                offset = y * vpr + xstart * self.planes
                end_offset = (y + 1) * vpr
                skip = self.planes * xstep
                for i in range(self.planes):
                    a[offset + i:end_offset:skip] = flat[i::self.planes]
    return a