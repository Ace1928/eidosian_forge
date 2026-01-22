import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def read_until_whitespace(stream: StreamType, maxchars: Optional[int]=None) -> bytes:
    """
    Read non-whitespace characters and return them.

    Stops upon encountering whitespace or when maxchars is reached.

    Args:
        stream: The data stream from which was read.
        maxchars: The maximum number of bytes returned; by default unlimited.

    Returns:
        The data which was read.
    """
    txt = b''
    while True:
        tok = stream.read(1)
        if tok.isspace() or not tok:
            break
        txt += tok
        if len(txt) == maxchars:
            break
    return txt