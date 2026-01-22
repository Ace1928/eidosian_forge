import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def undo_filter(self, filter_type, scanline, previous):
    """
        Undo the filter for a scanline.
        `scanline` is a sequence of bytes that
        does not include the initial filter type byte.
        `previous` is decoded previous scanline
        (for straightlaced images this is the previous pixel row,
        but for interlaced images, it is
        the previous scanline in the reduced image,
        which in general is not the previous pixel row in the final image).
        When there is no previous scanline
        (the first row of a straightlaced image,
        or the first row in one of the passes in an interlaced image),
        then this argument should be ``None``.

        The scanline will have the effects of filtering removed;
        the result will be returned as a fresh sequence of bytes.
        """
    result = scanline
    if filter_type == 0:
        return result
    if filter_type not in (1, 2, 3, 4):
        raise FormatError('Invalid PNG Filter Type.  See http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters .')
    fu = max(1, self.psize)
    if not previous:
        previous = bytearray([0] * len(scanline))
    fn = (None, undo_filter_sub, undo_filter_up, undo_filter_average, undo_filter_paeth)[filter_type]
    fn(fu, scanline, previous, result)
    return result