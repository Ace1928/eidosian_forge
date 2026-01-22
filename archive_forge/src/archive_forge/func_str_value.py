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
def str_value(x: object) -> bytes:
    x = resolve1(x)
    if not isinstance(x, bytes):
        if settings.STRICT:
            raise PDFTypeError('String required: %r' % x)
        return b''
    return x