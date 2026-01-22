import base64
import imghdr
from collections import OrderedDict
from os import path
from typing import IO, BinaryIO, NamedTuple, Optional, Tuple
import imagesize
def guess_mimetype(filename: str='', default: Optional[str]=None) -> Optional[str]:
    _, ext = path.splitext(filename.lower())
    if ext in mime_suffixes:
        return mime_suffixes[ext]
    elif path.exists(filename):
        with open(filename, 'rb') as f:
            return guess_mimetype_for_stream(f, default=default)
    return default