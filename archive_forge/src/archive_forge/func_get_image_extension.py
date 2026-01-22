import base64
import imghdr
from collections import OrderedDict
from os import path
from typing import IO, BinaryIO, NamedTuple, Optional, Tuple
import imagesize
def get_image_extension(mimetype: str) -> Optional[str]:
    for ext, _mimetype in mime_suffixes.items():
        if mimetype == _mimetype:
            return ext
    return None