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
def write_packed(self, outfile, rows):
    """
        Write PNG file to `outfile`.
        `rows` should be an iterator that yields each packed row;
        a packed row being a sequence of packed bytes.

        The rows have a filter byte prefixed and
        are then compressed into one or more IDAT chunks.
        They are not processed any further,
        so if bitdepth is other than 1, 2, 4, 8, 16,
        the pixel values should have been scaled
        before passing them to this method.

        This method does work for interlaced images but it is best avoided.
        For interlaced images, the rows should be
        presented in the order that they appear in the file.
        """
    self.write_preamble(outfile)
    if self.compression is not None:
        compressor = zlib.compressobj(self.compression)
    else:
        compressor = zlib.compressobj()
    data = bytearray()
    i = -1
    for i, row in enumerate(rows):
        data.append(0)
        data.extend(row)
        if len(data) > self.chunk_limit:
            compressed = compressor.compress(data)
            if len(compressed):
                write_chunk(outfile, b'IDAT', compressed)
            data = bytearray()
    compressed = compressor.compress(bytes(data))
    flushed = compressor.flush()
    if len(compressed) or len(flushed):
        write_chunk(outfile, b'IDAT', compressed + flushed)
    write_chunk(outfile, b'IEND')
    return i + 1