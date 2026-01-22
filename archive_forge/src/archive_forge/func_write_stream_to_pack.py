import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def write_stream_to_pack(read, write, zstream, base_crc=None):
    """Copy a stream as read from read function, zip it, and write the result.
    Count the number of written bytes and return it
    :param base_crc: if not None, the crc will be the base for all compressed data
        we consecutively write and generate a crc32 from. If None, no crc will be generated
    :return: tuple(no bytes read, no bytes written, crc32) crc might be 0 if base_crc
        was false"""
    br = 0
    bw = 0
    want_crc = base_crc is not None
    crc = 0
    if want_crc:
        crc = base_crc
    while True:
        chunk = read(chunk_size)
        br += len(chunk)
        compressed = zstream.compress(chunk)
        bw += len(compressed)
        write(compressed)
        if want_crc:
            crc = crc32(compressed, crc)
        if len(chunk) != chunk_size:
            break
    compressed = zstream.flush()
    bw += len(compressed)
    write(compressed)
    if want_crc:
        crc = crc32(compressed, crc)
    return (br, bw, crc)