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
def dict_value(x: object) -> Dict[Any, Any]:
    x = resolve1(x)
    if not isinstance(x, dict):
        if settings.STRICT:
            logger.error('PDFTypeError : Dict required: %r', x)
            raise PDFTypeError('Dict required: %r' % x)
        return {}
    return x