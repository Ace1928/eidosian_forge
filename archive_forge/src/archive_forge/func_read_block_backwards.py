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
def read_block_backwards(stream: StreamType, to_read: int) -> bytes:
    """
    Given a stream at position X, read a block of size to_read ending at position X.

    This changes the stream's position to the beginning of where the block was
    read.

    Args:
        stream:
        to_read:

    Returns:
        The data which was read.
    """
    if stream.tell() < to_read:
        raise PdfStreamError('Could not read malformed PDF file')
    stream.seek(-to_read, SEEK_CUR)
    read = stream.read(to_read)
    stream.seek(-to_read, SEEK_CUR)
    return read