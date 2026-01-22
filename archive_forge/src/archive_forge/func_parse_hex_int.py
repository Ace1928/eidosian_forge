import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def parse_hex_int(s: str) -> int:
    """Parse a non-negative hexadecimal integer from a string."""
    if HEXDIGITS.fullmatch(s) is None:
        raise ValueError('not a hexadecimal integer: %r' % s)
    return int(s, 16)