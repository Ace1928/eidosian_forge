from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def read_32(fobj, start_length, size):
    """
    Read a 32bit RGB icon resource.  Seems to be either uncompressed or
    an RLE packbits-like scheme.
    """
    start, length = start_length
    fobj.seek(start)
    pixel_size = (size[0] * size[2], size[1] * size[2])
    sizesq = pixel_size[0] * pixel_size[1]
    if length == sizesq * 3:
        indata = fobj.read(length)
        im = Image.frombuffer('RGB', pixel_size, indata, 'raw', 'RGB', 0, 1)
    else:
        im = Image.new('RGB', pixel_size, None)
        for band_ix in range(3):
            data = []
            bytesleft = sizesq
            while bytesleft > 0:
                byte = fobj.read(1)
                if not byte:
                    break
                byte = byte[0]
                if byte & 128:
                    blocksize = byte - 125
                    byte = fobj.read(1)
                    for i in range(blocksize):
                        data.append(byte)
                else:
                    blocksize = byte + 1
                    data.append(fobj.read(blocksize))
                bytesleft -= blocksize
                if bytesleft <= 0:
                    break
            if bytesleft != 0:
                msg = f'Error reading channel [{repr(bytesleft)} left]'
                raise SyntaxError(msg)
            band = Image.frombuffer('L', pixel_size, b''.join(data), 'raw', 'L', 0, 1)
            im.im.putband(band.im, band_ix)
    return {'RGB': im}