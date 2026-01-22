import io
import logging
import sys
import zlib
from typing import (
from . import settings
from .ascii85 import ascii85decode
from .ascii85 import asciihexdecode
from .ccitt import ccittfaxdecode
from .lzw import lzwdecode
from .psparser import LIT
from .psparser import PSException
from .psparser import PSObject
from .runlength import rldecode
from .utils import apply_png_predictor
def decompress_corrupted(data: bytes) -> bytes:
    """Called on some data that can't be properly decoded because of CRC checksum
    error. Attempt to decode it skipping the CRC.
    """
    d = zlib.decompressobj()
    f = io.BytesIO(data)
    result_str = b''
    buffer = f.read(1)
    i = 0
    try:
        while buffer:
            result_str += d.decompress(buffer)
            buffer = f.read(1)
            i += 1
    except zlib.error:
        if i < len(data) - 3:
            logger.warning('Data-loss while decompressing corrupted data')
    return result_str