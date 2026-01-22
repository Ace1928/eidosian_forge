from typing import (
from types import TracebackType
import logging
import re
import sys
import blessed
from .formatstring import fmtstr, FmtStr
from .formatstringarray import FSArray
from .termhelpers import Cbreak
def retrying_read() -> str:
    while True:
        try:
            c = in_stream.read(1)
            if c == '':
                raise ValueError("Stream should be blocking - shouldn't return ''. Returned %r so far", (resp,))
            return c
        except OSError:
            logger.info('stdin.read(1) that should never error just errored.')
            continue