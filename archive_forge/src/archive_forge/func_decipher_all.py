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
def decipher_all(decipher: DecipherCallable, objid: int, genno: int, x: object) -> Any:
    """Recursively deciphers the given object."""
    if isinstance(x, bytes):
        if len(x) == 0:
            return x
        return decipher(objid, genno, x)
    if isinstance(x, list):
        x = [decipher_all(decipher, objid, genno, v) for v in x]
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = decipher_all(decipher, objid, genno, v)
    return x